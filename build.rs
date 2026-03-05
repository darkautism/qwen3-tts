fn main() {
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-changed=cpp/code_pred_ggml_shim.cpp");

    // GGML static libraries needed with ggml-backend or ggml-predictor feature
    #[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
    {
        use std::path::PathBuf;

        let manifest_dir =
            PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));
        let ggml_base = std::env::var("GGML_LIB_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| manifest_dir.join("third_party/qwen3-tts.cpp"));
        let ggml_lib = ggml_base.join("ggml/build/src");
        let tts_lib = ggml_base.join("build");
        let code_pred_lib = tts_lib.join("libcode_pred_ggml.a");

        if !code_pred_lib.exists() {
            if let Err(err) = ensure_ggml_stack(&manifest_dir, &ggml_base) {
                println!("cargo:warning={}", err);
            }
        }

        if code_pred_lib.exists() {
            println!("cargo:rustc-link-search=native={}", tts_lib.display());
            println!("cargo:rustc-link-search=native={}", ggml_lib.display());

            println!("cargo:rustc-link-lib=static=code_pred_ggml");
            println!("cargo:rustc-link-lib=static=tts_transformer");
            println!("cargo:rustc-link-lib=static=ggml");
            println!("cargo:rustc-link-lib=static=ggml-cpu");
            println!("cargo:rustc-link-lib=static=ggml-base");

            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=gomp");
            println!("cargo:rustc-link-lib=m");
        } else {
            println!("cargo:warning=GGML libs not found at {}, ggml-backend/ggml-predictor will not work. See README source-build steps.", tts_lib.display());
        }
    }
}

#[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
fn ensure_ggml_stack(manifest_dir: &std::path::Path, ggml_base: &std::path::Path) -> Result<(), String> {
    use std::path::PathBuf;
    use std::process::Command;

    let tts_lib = ggml_base.join("build");
    let code_pred_lib = tts_lib.join("libcode_pred_ggml.a");
    if code_pred_lib.exists() {
        return Ok(());
    }

    if !ggml_base.join("CMakeLists.txt").exists() {
        let mut submodule_cmd = Command::new("git");
        submodule_cmd
            .arg("-C")
            .arg(manifest_dir)
            .arg("submodule")
            .arg("update")
            .arg("--init")
            .arg("--recursive")
            .arg("third_party/qwen3-tts.cpp");
        let _ = run_command(&mut submodule_cmd, "initialize qwen3-tts.cpp submodule");

        if !ggml_base.exists() {
            if let Some(parent) = ggml_base.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create {} failed: {}", parent.display(), e))?;
            }
            let mut clone_cmd = Command::new("git");
            clone_cmd
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg("https://github.com/predict-woo/qwen3-tts.cpp")
                .arg(ggml_base);
            run_command(&mut clone_cmd, "clone qwen3-tts.cpp source")
                .map_err(|e| format!("{} (set GGML_LIB_DIR to override source path)", e))?;
        }
    }

    if !ggml_base.join("CMakeLists.txt").exists() {
        return Err(format!(
            "GGML source missing at {}. Set GGML_LIB_DIR or initialize third_party/qwen3-tts.cpp.",
            ggml_base.display()
        ));
    }

    if !ggml_base.join("ggml/CMakeLists.txt").exists() {
        let mut nested_submodule_cmd = Command::new("git");
        nested_submodule_cmd
            .arg("-C")
            .arg(ggml_base)
            .arg("submodule")
            .arg("update")
            .arg("--init")
            .arg("--recursive");
        run_command(&mut nested_submodule_cmd, "initialize ggml nested submodule")
            .map_err(|e| format!("{} (run git submodule update --init --recursive)", e))?;
    }

    let ggml_lib = ggml_base.join("ggml/build/src");
    let ggml_ready = ggml_lib.join("libggml.a").exists()
        && ggml_lib.join("libggml-cpu.a").exists()
        && ggml_lib.join("libggml-base.a").exists();
    if !ggml_ready {
        let ggml_src = ggml_base.join("ggml");
        let ggml_build = ggml_base.join("ggml/build");

        let mut ggml_config_cmd = Command::new("cmake");
        ggml_config_cmd
            .arg("-S")
            .arg(&ggml_src)
            .arg("-B")
            .arg(&ggml_build)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("-DBUILD_SHARED_LIBS=OFF")
            .arg("-DGGML_BUILD_TESTS=OFF")
            .arg("-DGGML_BUILD_EXAMPLES=OFF");
        run_command(&mut ggml_config_cmd, "configure ggml")
            .map_err(|e| format!("{} (install cmake and build-essential)", e))?;

        let mut ggml_build_cmd = Command::new("cmake");
        ggml_build_cmd
            .arg("--build")
            .arg(&ggml_build)
            .arg("-j4");
        run_command(&mut ggml_build_cmd, "build ggml static libraries")
            .map_err(|e| format!("{} (check compiler/openmp toolchain)", e))?;
    }

    let tts_transformer_lib = tts_lib.join("libtts_transformer.a");
    if !tts_transformer_lib.exists() {
        let mut tts_config_cmd = Command::new("cmake");
        tts_config_cmd
            .arg("-S")
            .arg(ggml_base)
            .arg("-B")
            .arg(&tts_lib)
            .arg("-DCMAKE_BUILD_TYPE=Release");
        run_command(&mut tts_config_cmd, "configure tts_transformer")
            .map_err(|e| format!("{} (verify qwen3-tts.cpp source tree)", e))?;

        let mut tts_build_cmd = Command::new("cmake");
        tts_build_cmd
            .arg("--build")
            .arg(&tts_lib)
            .arg("--target")
            .arg("tts_transformer")
            .arg("-j4");
        run_command(&mut tts_build_cmd, "build tts_transformer")
            .map_err(|e| format!("{} (check ggml CMake output)", e))?;
    }

    if !code_pred_lib.exists() {
        let shim_src = manifest_dir.join("cpp/code_pred_ggml_shim.cpp");
        if !shim_src.exists() {
            return Err(format!(
                "Missing GGML shim source: {}",
                shim_src.display()
            ));
        }

        std::fs::create_dir_all(&tts_lib)
            .map_err(|e| format!("create {} failed: {}", tts_lib.display(), e))?;
        let shim_obj = tts_lib.join("code_pred_ggml_shim.o");
        let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());

        let mut cxx_cmd = Command::new(cxx);
        cxx_cmd
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-I")
            .arg(ggml_base.join("src"))
            .arg("-I")
            .arg(ggml_base.join("ggml/include"))
            .arg("-c")
            .arg(&shim_src)
            .arg("-o")
            .arg(&shim_obj);
        run_command(&mut cxx_cmd, "compile code_pred_ggml shim")
            .map_err(|e| format!("{} (set CXX if compiler not found)", e))?;

        let mut ar_cmd = Command::new("ar");
        ar_cmd
            .arg("rcs")
            .arg(&code_pred_lib)
            .arg(&shim_obj);
        run_command(&mut ar_cmd, "archive libcode_pred_ggml.a")
            .map_err(|e| format!("{} (install binutils)", e))?;
    }

    let mut built_files = Vec::<PathBuf>::new();
    built_files.push(code_pred_lib);
    built_files.push(tts_transformer_lib);
    built_files.push(ggml_lib.join("libggml.a"));
    built_files.push(ggml_lib.join("libggml-cpu.a"));
    built_files.push(ggml_lib.join("libggml-base.a"));
    for file in built_files {
        if !file.exists() {
            return Err(format!("missing expected GGML artifact: {}", file.display()));
        }
    }

    Ok(())
}

#[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
fn run_command(cmd: &mut std::process::Command, step: &str) -> Result<(), String> {
    println!("cargo:warning={} => {:?}", step, cmd);
    let status = cmd
        .status()
        .map_err(|e| format!("{} failed to start: {}", step, e))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed with status {}", step, status))
    }
}
