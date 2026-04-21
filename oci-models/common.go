// Common code shared between the standalone CLI binary and the C shared library.
package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/docker/model-runner/cmd/cli/pkg/standalone"
	"github.com/docker/model-runner/pkg/distribution/distribution"
	"golang.org/x/term"
)

// ollamaPullStatus matches the NDJSON objects emitted by Ollama's /api/pull.
type ollamaPullStatus struct {
	Status    string `json:"status,omitempty"`
	Digest    string `json:"digest,omitempty"`
	Total     uint64 `json:"total,omitempty"`
	Completed uint64 `json:"completed,omitempty"`
	Error     string `json:"error,omitempty"`
}

// shortDigest shortens a SHA-256 digest for Ollama-style progress messages.
func shortDigest(digest string) string {
	const prefix = "sha256:"
	const shortLen = 12

	if len(digest) >= len(prefix)+shortLen && digest[:len(prefix)] == prefix {
		return digest[len(prefix) : len(prefix)+shortLen]
	}
	if len(digest) > shortLen {
		return digest[:shortLen]
	}
	return digest
}

// getStorePath mirrors envconfig.ModelsPath() from model-runner.
func getStorePath() (string, error) {
	if s := os.Getenv("MODELS_PATH"); s != "" {
		return s, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".docker", "models"), nil
}

func newClient() (*distribution.Client, error) {
	storePath, err := getStorePath()
	if err != nil {
		return nil, fmt.Errorf("store path: %w", err)
	}

	// Use a quiet logger so only errors surface.
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelWarn,
	}))

	return distribution.NewClient(
		distribution.WithStoreRootPath(storePath),
		distribution.WithLogger(logger),
	)
}

// stderrPrinter implements standalone.StatusPrinter writing to stderr.
// This is the same interface that Docker Model Runner's CLI uses.
type stderrPrinter struct{}

func (p *stderrPrinter) Printf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format, args...)
}

func (p *stderrPrinter) Println(args ...any) {
	fmt.Fprintln(os.Stderr, args...)
}

func (p *stderrPrinter) PrintErrf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format, args...)
}

func (p *stderrPrinter) Write(b []byte) (n int, err error) {
	return os.Stderr.Write(b)
}

func (p *stderrPrinter) GetFdInfo() (fd uintptr, isTerminal bool) {
	fd = os.Stderr.Fd()
	return fd, term.IsTerminal(int(fd))
}

// Compile-time check that stderrPrinter implements StatusPrinter.
var _ standalone.StatusPrinter = (*stderrPrinter)(nil)
