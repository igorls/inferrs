use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");

    // Statically link the CUDA runtime instead of hard-linking libcudart.so.
    // libcudart_static.a resolves the CUDA driver API (libcuda.so) via
    // dlopen/dlsym internally, so the final binary ends up with NO DT_NEEDED
    // entries for libcudart / libcuda / libcublas / libcurand — matching the
    // behaviour already achieved for cudarc via `fallback-dynamic-loading`.
    // This is what makes "brew install inferrs" viable as a single binary
    // that dlopens whatever CUDA libs are present at runtime (12.x, 13.x, …)
    // and falls back cleanly on systems without CUDA at all.
    //
    // Also expose the CUDA toolkit lib dir explicitly so the static archive
    // is discoverable on toolchains that don't add it by default (e.g. sbsa
    // cross-builds where libcudart.so is missing but libcudart_static.a is
    // present).
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        if is_target_msvc {
            println!("cargo:rustc-link-search=native={cuda_path}/lib/x64");
        } else {
            println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        }
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    if !is_target_msvc {
        // cudart_static uses dlopen and POSIX realtime clocks internally.
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=rt");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}
