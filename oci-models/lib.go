//go:build cshared

// C shared library entry point for OCI model operations.
// Built with: go build -buildmode=c-shared -tags cshared
package main

/*
#include <stdlib.h>

// Callback type for streaming pull progress.  Each invocation passes one
// NDJSON line (NUL-terminated) and the opaque context pointer the caller
// supplied to oci_pull_stream.
typedef void (*oci_progress_fn)(const char* line, void* ctx);

// C trampoline — cgo cannot call C function pointers directly from Go.
static inline void invoke_progress(oci_progress_fn fn, const char* line, void* ctx) {
	fn(line, ctx);
}
*/
import "C"

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"unsafe"

	"github.com/docker/model-runner/cmd/cli/desktop"
	"github.com/docker/model-runner/pkg/distribution/oci"
)

// lastError stores the most recent error message.  Protected by a mutex
// for safety, although FFI callers are expected to be single-threaded.
var (
	lastErr   string
	lastErrMu sync.Mutex
)

func setLastError(err error) {
	lastErrMu.Lock()
	defer lastErrMu.Unlock()
	if err != nil {
		lastErr = err.Error()
	} else {
		lastErr = ""
	}
}

// oci_pull pulls an OCI model and returns the bundle path.
// Progress is written to stderr.  Returns NULL on error (retrieve with
// oci_last_error).  Caller must free the returned string with oci_free_string.
//
//export oci_pull
func oci_pull(reference *C.char) *C.char {
	ref := C.GoString(reference)

	client, err := newClient()
	if err != nil {
		setLastError(fmt.Errorf("client: %w", err))
		return nil
	}

	// Create a pipe: PullModel writes JSON progress to pw,
	// DisplayProgress reads from pr and renders Docker-style progress bars.
	pr, pw := io.Pipe()

	type pullResult struct {
		msg      string
		progress bool
		err      error
	}
	done := make(chan pullResult, 1)

	go func() {
		msg, progress, err := desktop.DisplayProgress(pr, &stderrPrinter{})
		done <- pullResult{msg, progress, err}
	}()

	// Pull the model.
	pullErr := client.PullModel(context.Background(), ref, pw)
	pw.Close()

	// Wait for progress rendering to finish.
	result := <-done
	if pullErr != nil {
		setLastError(fmt.Errorf("pull: %w", pullErr))
		return nil
	}
	if result.err != nil {
		setLastError(fmt.Errorf("progress: %w", result.err))
		return nil
	}

	// Print final message from progress stream (e.g. "Model pulled successfully").
	if result.msg != "" {
		fmt.Fprintf(os.Stderr, "%s\n", result.msg)
	}

	// Ensure the bundle is created and return its root directory.
	bundle, err := client.GetBundle(ref)
	if err != nil {
		setLastError(fmt.Errorf("bundle: %w", err))
		return nil
	}

	setLastError(nil)
	return C.CString(bundle.RootDir())
}

// oci_pull_stream pulls an OCI model and streams Ollama-compatible NDJSON
// progress to the given callback.  Each callback invocation receives one
// complete NDJSON line.  Returns 0 on success, -1 on error.  On error the
// callback will have received an NDJSON object with an "error" field and
// oci_last_error will contain the message.
//
//export oci_pull_stream
func oci_pull_stream(reference *C.char, callback C.oci_progress_fn, ctx unsafe.Pointer) C.int {
	ref := C.GoString(reference)

	client, err := newClient()
	if err != nil {
		emitCallback(callback, ctx, ollamaPullStatus{Error: fmt.Sprintf("client: %v", err)})
		setLastError(fmt.Errorf("client: %w", err))
		return -1
	}

	pr, pw := io.Pipe()
	done := make(chan error, 1)

	go func() {
		done <- streamProgressToCallback(pr, callback, ctx)
	}()

	pullErr := client.PullModel(context.Background(), ref, pw)
	pw.Close()
	progressErr := <-done

	if pullErr != nil {
		emitCallback(callback, ctx, ollamaPullStatus{Error: fmt.Sprintf("pull: %v", pullErr)})
		setLastError(fmt.Errorf("pull: %w", pullErr))
		return -1
	}
	if progressErr != nil {
		emitCallback(callback, ctx, ollamaPullStatus{Error: fmt.Sprintf("progress: %v", progressErr)})
		setLastError(fmt.Errorf("progress: %w", progressErr))
		return -1
	}

	// Emit the final Ollama-compatible status sequence.
	for _, status := range []string{
		"verifying sha256 digest",
		"writing manifest",
		"removing any unused layers",
		"success",
	} {
		emitCallback(callback, ctx, ollamaPullStatus{Status: status})
	}

	setLastError(nil)
	return 0
}

// emitCallback serializes a pull status as JSON and invokes the C callback.
func emitCallback(callback C.oci_progress_fn, ctx unsafe.Pointer, status ollamaPullStatus) {
	data, err := json.Marshal(status)
	if err != nil {
		return
	}
	cLine := C.CString(string(data))
	C.invoke_progress(callback, cLine, ctx)
	C.free(unsafe.Pointer(cLine))
}

// streamProgressToCallback reads Docker Model Runner progress messages from r
// and emits Ollama-compatible NDJSON to the C callback.
func streamProgressToCallback(r io.Reader, callback C.oci_progress_fn, ctx unsafe.Pointer) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	emittedManifest := false
	for scanner.Scan() {
		var msg oci.ProgressMessage
		if err := json.Unmarshal(scanner.Bytes(), &msg); err != nil {
			return fmt.Errorf("decode progress: %w", err)
		}

		switch msg.Type {
		case oci.TypeProgress:
			if !emittedManifest {
				emitCallback(callback, ctx, ollamaPullStatus{Status: "pulling manifest"})
				emittedManifest = true
			}
			if msg.Layer.ID == "" {
				continue
			}
			emitCallback(callback, ctx, ollamaPullStatus{
				Status:    fmt.Sprintf("pulling %s", shortDigest(msg.Layer.ID)),
				Digest:    msg.Layer.ID,
				Total:     msg.Layer.Size,
				Completed: msg.Layer.Current,
			})
		case oci.TypeWarning:
			emitCallback(callback, ctx, ollamaPullStatus{Status: msg.Message})
		case oci.TypeError:
			return fmt.Errorf("%s", msg.Message)
		case oci.TypeSuccess:
			// Ignore; final sequence emitted after PullModel returns.
		default:
			return fmt.Errorf("unsupported progress message type %q", msg.Type)
		}
	}

	return scanner.Err()
}

// oci_bundle returns the bundle path for an already-pulled model.
// Returns NULL if the model is not in the local store (check oci_last_error
// for details).  Caller must free the returned string with oci_free_string.
//
//export oci_bundle
func oci_bundle(reference *C.char) *C.char {
	ref := C.GoString(reference)

	client, err := newClient()
	if err != nil {
		setLastError(fmt.Errorf("client: %w", err))
		return nil
	}

	bundle, err := client.GetBundle(ref)
	if err != nil {
		setLastError(fmt.Errorf("bundle: %w", err))
		return nil
	}

	setLastError(nil)
	return C.CString(bundle.RootDir())
}

// oci_list returns all local OCI models as tab-separated "tag\tid\n" lines.
// Returns NULL on error (check oci_last_error for details).  Caller must free
// the returned string with oci_free_string.
//
//export oci_list
func oci_list() *C.char {
	client, err := newClient()
	if err != nil {
		setLastError(fmt.Errorf("client: %w", err))
		return nil
	}

	models, err := client.ListModels()
	if err != nil {
		setLastError(fmt.Errorf("list: %w", err))
		return nil
	}

	var buf strings.Builder
	for _, m := range models {
		id, err := m.ID()
		if err != nil {
			continue
		}
		for _, tag := range m.Tags() {
			fmt.Fprintf(&buf, "%s\t%s\n", tag, id)
		}
	}

	setLastError(nil)
	return C.CString(buf.String())
}

// oci_last_error returns the last error message, or NULL if no error.
// The returned string must be freed by the caller with oci_free_string.
//
//export oci_last_error
func oci_last_error() *C.char {
	lastErrMu.Lock()
	defer lastErrMu.Unlock()
	if lastErr == "" {
		return nil
	}
	return C.CString(lastErr)
}

// oci_free_string frees a string previously returned by oci_pull, oci_bundle,
// oci_list, or oci_last_error.
//
//export oci_free_string
func oci_free_string(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

// Empty main required by -buildmode=c-shared.
func main() {}
