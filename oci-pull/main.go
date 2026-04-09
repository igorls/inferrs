// inferrs-oci-pull — OCI model pull helper for inferrs.
//
// Uses Docker Model Runner's distribution library (containerd transport)
// to pull models from OCI registries.  Shares the same store as DMR at
// ~/.docker/models so models already pulled by DMR are reused and vice versa.
//
// Commands:
//
//	inferrs-oci-pull pull  <reference>   Pull a model and print the bundle path.
//	inferrs-oci-pull bundle <reference>  Print the bundle path for an already-pulled model.
//	inferrs-oci-pull list                List all models in the store (tag\tid).
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/docker/model-runner/pkg/distribution/distribution"
)

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

func cmdPull(ref string) error {
	client, err := newClient()
	if err != nil {
		return err
	}

	// Pull the model (progress goes to stderr).
	if err := client.PullModel(context.Background(), ref, os.Stderr); err != nil {
		return fmt.Errorf("pull: %w", err)
	}

	// Ensure the bundle is created and print its root directory.
	bundle, err := client.GetBundle(ref)
	if err != nil {
		return fmt.Errorf("bundle: %w", err)
	}

	// stdout: the bundle path — this is what the Rust caller reads.
	fmt.Println(bundle.RootDir())
	return nil
}

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

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-pull <command> [args]\n\n")
	fmt.Fprintf(os.Stderr, "Commands:\n")
	fmt.Fprintf(os.Stderr, "  pull  <reference>   Pull a model and print bundle path\n")
	fmt.Fprintf(os.Stderr, "  bundle <reference>  Print bundle path for existing model\n")
	fmt.Fprintf(os.Stderr, "  list                List all models in the store\n")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "pull":
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-pull pull <reference>\n")
			os.Exit(1)
		}
		err = cmdPull(os.Args[2])
	case "bundle":
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-pull bundle <reference>\n")
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
