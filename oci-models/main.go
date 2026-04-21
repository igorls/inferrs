//go:build !cshared

// inferrs-oci-models — OCI model helper for inferrs (standalone CLI).
//
// Uses Docker Model Runner's distribution library (containerd transport)
// to pull models from OCI registries.  Shares the same store as DMR at
// ~/.docker/models so models already pulled by DMR are reused and vice versa.
//
// Commands:
//
//	inferrs-oci-models pull   <reference>   Pull a model and print the bundle path.
//	inferrs-oci-models stream <reference>   Pull a model and emit Ollama NDJSON progress.
//	inferrs-oci-models bundle <reference>   Print the bundle path for an already-pulled model.
//	inferrs-oci-models list                 List all models in the store (tag\tid).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/docker/model-runner/cmd/cli/desktop"
	"github.com/docker/model-runner/pkg/distribution/oci"
)

// cmdPull pulls a model and prints its bundle directory to stdout.
func cmdPull(ref string) error {
	client, err := newClient()
	if err != nil {
		return err
	}

	// Create a pipe: PullModel writes JSON progress to pw, and DMR's progress
	// printer renders Docker-style progress bars from pr.
	pr, pw := io.Pipe()

	type pullResult struct {
		msg string
		err error
	}
	done := make(chan pullResult, 1)

	go func() {
		msg, _, err := desktop.DisplayProgress(pr, &stderrPrinter{})
		done <- pullResult{msg: msg, err: err}
	}()

	// Pull the model and wait until the progress renderer drains the pipe.
	pullErr := client.PullModel(context.Background(), ref, pw)
	_ = pw.Close()
	result := <-done
	if pullErr != nil {
		return fmt.Errorf("pull: %w", pullErr)
	}
	if result.err != nil {
		return fmt.Errorf("progress: %w", result.err)
	}

	// Print the final helper message, if any, before returning the bundle path.
	if result.msg != "" {
		fmt.Fprintf(os.Stderr, "%s\n", result.msg)
	}

	bundle, err := client.GetBundle(ref)
	if err != nil {
		return fmt.Errorf("bundle: %w", err)
	}

	fmt.Println(bundle.RootDir())
	return nil
}

// cmdStream pulls a model and emits Ollama-compatible NDJSON progress.
func cmdStream(ref string) error {
	client, err := newClient()
	if err != nil {
		return err
	}

	pr, pw := io.Pipe()
	done := make(chan error, 1)

	go func() {
		done <- forwardPullProgress(pr)
	}()

	pullErr := client.PullModel(context.Background(), ref, pw)
	_ = pw.Close()
	progressErr := <-done

	if pullErr != nil {
		return writeStreamError(fmt.Errorf("pull: %w", pullErr))
	}
	if progressErr != nil {
		return writeStreamError(fmt.Errorf("progress: %w", progressErr))
	}

	for _, status := range []string{
		"verifying sha256 digest",
		"writing manifest",
		"removing any unused layers",
		"success",
	} {
		if err := writePullStatus(ollamaPullStatus{Status: status}); err != nil {
			return err
		}
	}

	return nil
}

// cmdBundle prints the bundle directory of an already-pulled model.
func cmdBundle(ref string) error {
	client, err := newClient()
	if err != nil {
		return err
	}

	bundle, err := client.GetBundle(ref)
	if err != nil {
		return fmt.Errorf("bundle: %w", err)
	}

	fmt.Println(bundle.RootDir())
	return nil
}

// cmdList prints local OCI models in a simple tab-separated format.
func cmdList() error {
	client, err := newClient()
	if err != nil {
		return err
	}

	models, err := client.ListModels()
	if err != nil {
		return fmt.Errorf("list: %w", err)
	}

	for _, m := range models {
		id, err := m.ID()
		if err != nil {
			continue
		}
		for _, tag := range m.Tags() {
			fmt.Printf("%s\t%s\n", tag, id)
		}
	}
	return nil
}

// forwardPullProgress converts DMR OCI progress messages into Ollama NDJSON.
func forwardPullProgress(r io.Reader) error {
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
				if err := writePullStatus(
					ollamaPullStatus{Status: "pulling manifest"},
				); err != nil {
					return err
				}
				emittedManifest = true
			}
			if msg.Layer.ID == "" {
				continue
			}
			if err := writePullStatus(ollamaPullStatus{
				Status:    fmt.Sprintf("pulling %s", shortDigest(msg.Layer.ID)),
				Digest:    msg.Layer.ID,
				Total:     msg.Layer.Size,
				Completed: msg.Layer.Current,
			}); err != nil {
				return err
			}
		case oci.TypeWarning:
			if err := writePullStatus(
				ollamaPullStatus{Status: msg.Message},
			); err != nil {
				return err
			}
		case oci.TypeError:
			return fmt.Errorf("%s", msg.Message)
		case oci.TypeSuccess:
			// Ignore the raw success event. The helper emits the final
			// Ollama-compatible completion sequence after PullModel returns.
		default:
			return fmt.Errorf("unsupported progress message type %q", msg.Type)
		}
	}

	return scanner.Err()
}

// writePullStatus writes one Ollama pull status object to stdout.
func writePullStatus(status ollamaPullStatus) error {
	return json.NewEncoder(os.Stdout).Encode(status)
}

// writeStreamError writes an Ollama-compatible error object and returns it.
func writeStreamError(err error) error {
	if err == nil {
		return nil
	}
	_ = writePullStatus(ollamaPullStatus{Error: err.Error()})
	return err
}

// usage prints CLI usage help to stderr.
func usage() {
	fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-models <command> [args]\n\n")
	fmt.Fprintf(os.Stderr, "Commands:\n")
	fmt.Fprintf(
		os.Stderr,
		"  pull   <reference>   Pull a model and print bundle path\n",
	)
	fmt.Fprintf(
		os.Stderr,
		"  stream <reference>   Pull a model and emit Ollama NDJSON progress\n",
	)
	fmt.Fprintf(
		os.Stderr,
		"  bundle <reference>   Print bundle path for existing model\n",
	)
	fmt.Fprintf(os.Stderr, "  list                 List all models in the store\n")
}

// main dispatches the requested helper subcommand.
func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "pull":
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-models pull <reference>\n")
			os.Exit(1)
		}
		err = cmdPull(os.Args[2])
	case "stream":
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-models stream <reference>\n")
			os.Exit(1)
		}
		err = cmdStream(os.Args[2])
	case "bundle":
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-models bundle <reference>\n")
			os.Exit(1)
		}
		err = cmdBundle(os.Args[2])
	case "list":
		err = cmdList()
	default:
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
