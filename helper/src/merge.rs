use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use syn::Item;

use crate::ModuleConfig;

#[derive(Debug, Ord, PartialEq, PartialOrd, Eq, Clone, Copy)]
struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl std::fmt::Display for Version {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug)]
struct FunctionInfo {
    declarations: BTreeMap<Version, String>, // version -> declaration
}

#[derive(Default)]
struct BindingMerger {
    functions: BTreeMap<String, FunctionInfo>,
    ftypes: BTreeMap<String, FunctionInfo>,
    fconsts: BTreeMap<String, FunctionInfo>,
    enums: BTreeMap<String, FunctionInfo>,
    impls: BTreeMap<String, FunctionInfo>,
    structs: BTreeMap<String, FunctionInfo>,
    types: BTreeMap<String, FunctionInfo>,
    uses: BTreeMap<String, FunctionInfo>,
    unions: BTreeMap<String, FunctionInfo>,
    consts: BTreeMap<String, FunctionInfo>,
}

impl BindingMerger {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn process_file(&mut self, path: &Path, version: &Version) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let file = syn::parse_file(&content)?;

        for item in file.items {
            match item {
                Item::ForeignMod(foreign_mod) => {
                    for item in foreign_mod.items {
                        if let syn::ForeignItem::Fn(func) = &item {
                            let name = func.sig.ident.to_string();
                            let declaration = quote::quote!(#func).to_string();

                            let entry =
                                self.functions.entry(name).or_insert_with(|| FunctionInfo {
                                    declarations: BTreeMap::new(),
                                });

                            entry.declarations.insert(version.clone(), declaration);
                        }
                        if let syn::ForeignItem::Type(t) = &item {
                            let name = t.ident.to_string();
                            let declaration = quote::quote!(#t).to_string();

                            let entry = self.ftypes.entry(name).or_insert_with(|| FunctionInfo {
                                declarations: BTreeMap::new(),
                            });

                            entry.declarations.insert(version.clone(), declaration);
                        }
                        if let syn::ForeignItem::Static(constant) = &item {
                            let name = constant.ident.to_string();
                            let declaration = quote::quote!(#constant).to_string();

                            let entry = self.fconsts.entry(name).or_insert_with(|| FunctionInfo {
                                declarations: BTreeMap::new(),
                            });

                            entry.declarations.insert(version.clone(), declaration);
                        }
                    }
                }
                Item::Struct(st) => {
                    let name = st.ident.to_string();
                    let declaration = quote::quote!(#st).to_string();

                    let entry = self.structs.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), declaration);
                }
                Item::Type(typ) => {
                    let name = typ.ident.to_string();
                    let declaration = quote::quote!(#typ).to_string();

                    let entry = self.types.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), declaration);
                }
                Item::Impl(imp) => {
                    let name = format!("{:?}", (*imp.self_ty));
                    let declaration = quote::quote!(#imp).to_string();

                    let entry = self.impls.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), declaration);
                }
                Item::Enum(en) => {
                    let name = en.ident.to_string();
                    let declaration = quote::quote!(#en).to_string();

                    let entry = self.enums.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), declaration);
                }
                Item::Use(us) => {
                    let name = format!("{us:?}");
                    let declaration = quote::quote!(#us).to_string();

                    let entry = self.uses.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), declaration);
                }
                Item::Union(un) => {
                    let name = un.ident.to_string();
                    let declaration = quote::quote!(#un).to_string();

                    let entry = self.unions.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), declaration);
                }
                Item::Const(con) => {
                    let name = con.ident.to_string();
                    let declaration = quote::quote!(#con).to_string();

                    let entry = self.consts.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), declaration);
                }
                other_item => {
                    println!("Ignored item {other_item:?}");
                }
            }
        }

        Ok(())
    }

    pub fn generate_unified_bindings(&self) -> String {
        let mut output = String::new();

        output.push_str("// AUTOGENERATED UNIFIED CUDA BINDINGS\n");
        output.push_str("// This file combines bindings from multiple CUDA versions\n\n");
        output.push_str("#![allow(non_camel_case_types)]\n");
        output.push_str("#![allow(non_snake_case)]\n");
        output.push_str("#![allow(dead_code)]\n\n");

        output.push_str("extern \"C\" {\n");
        write_to_output(&self.functions, &mut output).expect("Write to output");
        write_to_output(&self.ftypes, &mut output).expect("Write to output");
        write_to_output(&self.fconsts, &mut output).expect("Write to output");
        output.push_str("}\n");

        write_to_output(&self.enums, &mut output).expect("Write to output");
        write_to_output(&self.impls, &mut output).expect("Write to output");
        write_to_output(&self.structs, &mut output).expect("Write to output");
        write_to_output(&self.types, &mut output).expect("Write to output");
        write_to_output(&self.uses, &mut output).expect("Write to output");
        write_to_output(&self.unions, &mut output).expect("Write to output");
        write_to_output(&self.consts, &mut output).expect("Write to output");

        // Add similar sections for types and constants

        output
    }
}

fn write_to_output(info: &BTreeMap<String, FunctionInfo>, output: &mut String) -> Result<()> {
    for (name, info) in info {
        // Function with version-specific declarations
        let mut prev_decl = None;
        let mut versions = vec![];
        for (version, decl) in &info.declarations {
            if let Some(prev_decl) = prev_decl {
                if prev_decl == decl {
                    versions.push(version);
                    continue;
                } else {
                    if !versions.is_empty() {
                        println!("Breaking change detected in {version} for {name}");
                    }
                    versions.push(version);
                    let features = versions
                        .iter()
                        .map(|v| format!("feature = \"{}\"", version_to_feature(v)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    output.push_str(&format!("#[cfg(any({features}))]\n"));
                    output.push_str(prev_decl);
                    output.push_str("\n");
                    versions.clear();
                }
            } else {
                versions.push(version);
            }
            prev_decl = Some(decl);
        }
        if !versions.is_empty() {
            if let Some(decl) = prev_decl {
                let features = versions
                    .into_iter()
                    .map(|v| format!("feature = \"{}\"", version_to_feature(v)))
                    .collect::<Vec<_>>()
                    .join(", ");
                output.push_str(&format!("#[cfg(any({features}))]\n"));
                output.push_str(decl);
                output.push_str("\n");
            }
        }
    }
    Ok(())
}

fn version_to_feature(version: &Version) -> String {
    format!(
        "cuda-{:0>2}{:0>2}{}",
        version.major, version.minor, version.patch
    )
}

pub fn merge<P: AsRef<Path>>(binding_dir: P, output_filename: P) -> Result<()> {
    let mut merger = BindingMerger::new();

    let entries = fs::read_dir(binding_dir.as_ref())?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Ok(version) = extract_version_from_filename(&path.display().to_string()) {
                merger.process_file(&path, &version)?;
            }
        }
    }

    // Generate unified output
    let unified = merger.generate_unified_bindings();
    std::fs::write(output_filename, unified)?;

    // Cleanup old files
    let entries = fs::read_dir(binding_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Ok(_) = extract_version_from_filename(&path.display().to_string()) {
                std::fs::remove_file(path)?;
            }
        }
    }

    Ok(())
}

pub fn merge_bindings(modules: &[(String, ModuleConfig)]) -> Result<()> {
    for (name, _config) in modules {
        merge(
            format!("../src/{name}/sys/linked"),
            format!("../src/{name}/sys/linked/unified.rs"),
        )?;
    }
    Ok(())
}

fn extract_version_from_filename(cuda_version: &str) -> Result<Version> {
    let number = cuda_version
        .split('_')
        .last()
        .context(format!("Invalid CUDA version format: {}", cuda_version))?;

    let major = number[..2].parse().context(format!(
        "Failed to parse major version from {}",
        cuda_version
    ))?;
    let minor = number[2..4].parse().context(format!(
        "Failed to parse minor version from {}",
        cuda_version
    ))?;
    let patch = number[4..5].parse().context(format!(
        "Failed to parse patch version from {}",
        cuda_version
    ))?;

    Ok(Version {
        major,
        minor,
        patch,
    })
}
