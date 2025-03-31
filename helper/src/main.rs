use clap::{Arg, Command};
use quote::quote;
use std::fs;
use syn::{parse_file, FnArg, ForeignItem, Pat};

fn main() {
    let matches = Command::new("adapt_libloading")
        .version("1.0")
        .author("Your Name")
        .about("Converts extern C bindings to Rust dynamic loading bindings")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("FILE")
                .help("Sets the input file to read")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Sets the output file to write")
                .required(true),
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();

    let file_content = fs::read_to_string(input_path).expect("Failed to read input file");

    let parsed_file = parse_file(&file_content).expect("Failed to parse Rust code from input file");

    let functions = parsed_file
        .items
        .into_iter()
        .filter_map(|item| {
            if let syn::Item::ForeignMod(foreign_mod) = item {
                Some(foreign_mod.items.into_iter().filter_map(|fitem| {
                    if let ForeignItem::Fn(func) = fitem {
                        let fn_name = &func.sig.ident;
                        let inputs = &func.sig.inputs;
                        let output = &func.sig.output;
                        // Extract only argument names (without types)
                        let arg_names = inputs.iter().filter_map(|arg| {
                            if let FnArg::Typed(pat_type) = arg {
                                if let Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                                    return Some(pat_ident.ident.clone());
                                }
                            }
                            None
                        });

                        Some(quote! {
                            pub unsafe fn #fn_name(#inputs) #output {
                                unsafe {
                                    culib().#fn_name(#(#arg_names),*)
                                }
                            }
                        })
                    } else {
                        None
                    }
                }))
            } else {
                None
            }
        })
        .flatten();

    let mut output_code = String::new();
    output_code.push_str(
        &quote! {
            use super::*;
        }
        .to_string(),
    );
    for function in functions {
        output_code.push_str(&function.to_string());
        output_code.push_str("\n");
    }

    fs::write(output_path, output_code).expect("Failed to write output file");

    println!("Successfully generated Rust bindings in {}", output_path);
}
