fn main() {
    // GGML static libraries only needed with ggml-backend feature
    #[cfg(feature = "ggml-backend")]
    {
        let ggml_base = std::env::var("GGML_LIB_DIR").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/home/kautism".to_string());
            format!("{}/qwen3_research/qwen3-tts.cpp", home)
        });

        let ggml_lib = format!("{}/ggml/build/src", ggml_base);
        let tts_lib = format!("{}/build", ggml_base);

        if std::path::Path::new(&format!("{}/libcode_pred_ggml.a", tts_lib)).exists() {
            println!("cargo:rustc-link-search=native={}", tts_lib);
            println!("cargo:rustc-link-search=native={}", ggml_lib);

            println!("cargo:rustc-link-lib=static=code_pred_ggml");
            println!("cargo:rustc-link-lib=static=tts_transformer");
            println!("cargo:rustc-link-lib=static=ggml");
            println!("cargo:rustc-link-lib=static=ggml-cpu");
            println!("cargo:rustc-link-lib=static=ggml-base");

            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=gomp");
            println!("cargo:rustc-link-lib=m");
        } else {
            println!(
                "cargo:warning=GGML libs not found at {}, ggml-backend will not work",
                tts_lib
            );
        }
    }
}
