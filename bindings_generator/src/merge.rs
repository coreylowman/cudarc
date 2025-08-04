use anyhow::{Context, Result};
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use syn::parse::Parser;
use syn::{
    Expr, Field, FnArg, ForeignItemFn, Item, ItemConst, ItemEnum, ItemFn, ItemImpl, ItemStruct,
    ItemType, ItemUnion, ItemUse, Pat, Stmt,
};

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
        // is very similar to `log::debug!`.
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

struct LibItem {
    adapter_function: ItemFn,
    member: Field,
    init_member: Stmt,
    init_decl: Expr,
}
struct LibItems {
    adapter_functions: Vec<ItemFn>,
    members: Vec<Field>,
    init_members: Vec<Stmt>,
    init_fields: Vec<Expr>,
}
impl LibItem {
    fn new(func: &ForeignItemFn, versions: &[&Version], n_versions: usize) -> Self {
        let parser = Field::parse_named;
        let features = versions
            .iter()
            .map(|v| version_to_feature(v))
            .collect::<Vec<_>>();
        let feature_tok = if versions.len() == n_versions {
            quote! {}
        } else {
            quote! {
                #[cfg(any(#(feature=#features),*))]
            }
        };
        let ForeignItemFn {
            attrs: _,
            vis: _,
            sig,
            semi_token: _,
        } = func;
        let fn_name = &sig.ident;
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

        let args = arg_names;
        let c = quote! {
            #feature_tok
            pub unsafe fn #fn_name(#inputs) #output{
                (culib().#fn_name)(#(#args),*)
            }
        };
        let adapter_function: ItemFn = syn::parse2(c.clone()).unwrap();
        let symbol_cstr = cstr_expr(fn_name.to_string());
        let init_member = syn::parse2(quote! {
            #feature_tok
            let #fn_name = __library
                .get(#symbol_cstr)
                .map(|sym| *sym).expect("Expected symbol in library");
        })
        .unwrap();
        let init_decl = syn::parse2(quote! {
            #feature_tok
            #fn_name
        })
        .unwrap();
        let c = quote! {
            #feature_tok
            pub #fn_name: unsafe extern "C" fn(#inputs) #output
        };
        let member = parser.parse2(c).unwrap();
        Self {
            adapter_function,
            init_member,
            init_decl,
            member,
        }
    }
}

pub fn cstr_expr(mut string: String) -> TokenStream {
    string.push('\0');
    let b = proc_macro2::Literal::byte_string(string.as_bytes());
    quote! {
        #b
    }
}

impl From<Vec<LibItem>> for LibItems {
    fn from(value: Vec<LibItem>) -> Self {
        let (adapter_functions, members, init_members, init_fields) = value
            .into_iter()
            .map(|v| (v.adapter_function, v.member, v.init_member, v.init_decl))
            .collect();
        Self {
            adapter_functions,
            members,
            init_members,
            init_fields,
        }
    }
}

#[derive(Debug)]
struct FunctionInfo<T> {
    declarations: BTreeMap<Version, T>, // version -> declaration
}

#[derive(Default)]
struct BindingMerger {
    functions: BTreeMap<String, FunctionInfo<ForeignItemFn>>,
    enums: BTreeMap<String, FunctionInfo<ItemEnum>>,
    impls: BTreeMap<String, FunctionInfo<ItemImpl>>,
    structs: BTreeMap<String, FunctionInfo<ItemStruct>>,
    types: BTreeMap<String, FunctionInfo<ItemType>>,
    uses: BTreeMap<String, FunctionInfo<ItemUse>>,
    unions: BTreeMap<String, FunctionInfo<ItemUnion>>,
    consts: BTreeMap<String, FunctionInfo<ItemConst>>,

    lib_names: Vec<String>,
    n_versions: usize,
}

impl BindingMerger {
    pub fn new(lib_names: Vec<String>) -> Self {
        Self {
            lib_names,
            n_versions: 0,
            ..Default::default()
        }
    }

    pub fn process_file(&mut self, path: &Path, version: &Version) -> Result<()> {
        self.n_versions += 1;
        let content = std::fs::read_to_string(path)?;
        let file = syn::parse_file(&content)?;

        for item in file.items {
            match item {
                Item::ForeignMod(foreign_mod) => {
                    for item in foreign_mod.items {
                        match &item {
                            syn::ForeignItem::Fn(func) => {
                                let name = func.sig.ident.to_string();
                                // let declaration = quote::quote!(#func).to_string();

                                let entry =
                                    self.functions.entry(name).or_insert_with(|| FunctionInfo {
                                        declarations: BTreeMap::new(),
                                    });

                                entry.declarations.insert(version.clone(), func.clone());
                            }
                            other => panic!("Unhandled foreign item {other:?}"),
                        }
                    }
                }
                Item::Struct(st) => {
                    let name = st.ident.to_string();
                    let entry = self.structs.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), st);
                }
                Item::Type(typ) => {
                    let name = typ.ident.to_string();

                    let entry = self.types.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), typ);
                }
                Item::Impl(imp) => {
                    let name = format!("{:?}", imp);

                    let entry = self.impls.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), imp);
                }
                Item::Enum(en) => {
                    let name = en.ident.to_string();

                    let entry = self.enums.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), en);
                }
                Item::Use(us) => {
                    let name = format!("{us:?}");

                    let entry = self.uses.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), us);
                }
                Item::Union(un) => {
                    let name = un.ident.to_string();
                    let entry = self.unions.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), un);
                }
                Item::Const(con) => {
                    let name = con.ident.to_string();

                    let entry = self.consts.entry(name).or_insert_with(|| FunctionInfo {
                        declarations: BTreeMap::new(),
                    });

                    entry.declarations.insert(version.clone(), con);
                }
                other_item => {
                    panic!("Unhandled item {other_item:?}");
                }
            }
        }

        Ok(())
    }

    pub fn generate_unified_bindings(&self) -> TokenStream {
        let enums = self.write_to_output(&self.enums).expect("Write to output");
        let impls = self.write_to_output(&self.impls).expect("Write to output");
        let structs = self
            .write_to_output(&self.structs)
            .expect("Write to output");
        let types = self.write_to_output(&self.types).expect("Write to output");
        let uses = self.write_to_output(&self.uses).expect("Write to output");
        let unions = self.write_to_output(&self.unions).expect("Write to output");
        let consts = self.write_to_output(&self.consts).expect("Write to output");
        let functions = self
            .write_to_output(&self.functions)
            .expect("Write to output");

        let lib_names = &self.lib_names;

        let loading_lib = self
            .create_loading_lib(&self.functions)
            .expect("Write to output");

        TokenStream::from(quote! {
            // AUTOGENERATED UNIFIED CUDA BINDINGS
            // This file combines bindings from multiple CUDA versions
            #![cfg_attr(feature = "no-std", no_std)]
            #![allow(non_camel_case_types)]
            #![allow(non_snake_case)]
            #![allow(dead_code)]

            #[cfg(feature = "no-std")]
            extern crate alloc;
            #[cfg(feature = "no-std")]
            extern crate no_std_compat as std;

            #uses

            #consts

            #types

            #enums

            #structs

            #impls

            #unions

            #[cfg(not(feature="dynamic-loading"))]
            extern "C" {
                #functions
            }

            #[cfg(feature="dynamic-loading")]
            mod loaded{
               use super::*;

               #loading_lib

               pub unsafe fn is_culib_present() -> bool {
                   let lib_names = [#(#lib_names),*];
                   let choices = lib_names
                       .iter()
                       .map(|l| crate::get_lib_name_candidates(l))
                       .flatten();
                   for choice in choices {
                       if Lib::new(choice).is_ok() {
                           return true;
                       }
                   }
                   false
               }

               pub unsafe fn culib() -> &'static Lib {
                   static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
                   LIB.get_or_init(|| {
                       let lib_names = std::vec![#(#lib_names),*];
                       let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
                       for choice in choices.iter() {
                           if let Ok(lib) = Lib::new(choice) {
                               return lib;
                           }
                       }
                       crate::panic_no_lib_found(lib_names[0], &choices);
                   })
               }

            }
            #[cfg(feature="dynamic-loading")]
            pub use loaded::*;
        })
    }

    fn write_to_output<T: ToTokens + PartialEq<T>>(
        &self,
        info: &BTreeMap<String, FunctionInfo<T>>,
    ) -> Result<TokenStream> {
        let mut output = TokenStream::new();
        for (name, info) in info {
            // Function with version-specific declarations
            let mut prev_decl: Option<&T> = None;
            let mut versions = vec![];
            for (version, decl) in &info.declarations {
                if let Some(prev_decl) = prev_decl {
                    if prev_decl != decl {
                        if !versions.is_empty() {
                            log::debug!("Breaking change detected in {version} for {name}");
                        }
                        let features = versions
                            .iter()
                            .map(|v| version_to_feature(v))
                            .collect::<Vec<_>>();
                        output.extend(quote! {
                            #[cfg(any(#(feature = #features), *))]
                            #prev_decl
                        });
                        versions.clear();
                    }
                }
                versions.push(*version);
                prev_decl = Some(decl.into());
            }
            if !versions.is_empty() {
                if let Some(decl) = prev_decl {
                    if versions.len() == self.n_versions {
                        // XXX small nicety, if every single version implements
                        // a function, just remove the feature flags.
                        output.extend(decl.into_token_stream());
                    } else {
                        let features = versions
                            .iter()
                            .map(|v| version_to_feature(v))
                            .collect::<Vec<_>>();
                        output.extend(quote! {
                            #[cfg(any(#(feature = #features),*))]
                            #decl
                        });
                    }
                } else {
                    panic!("Previous version shouldn't be empty");
                }
            } else {
                panic!("Versions shouldn't be empty");
            }
        }
        Ok(output)
    }

    fn create_loading_lib(
        &self,
        info: &BTreeMap<String, FunctionInfo<ForeignItemFn>>,
    ) -> Result<TokenStream> {
        let mut elements = vec![];
        for (_name, info) in info {
            // Function with version-specific declarations
            let mut prev_decl: Option<&ForeignItemFn> = None;
            let mut versions = vec![];
            for (version, decl) in &info.declarations {
                if let Some(prev_decl) = prev_decl {
                    if prev_decl != decl {
                        let element = LibItem::new(prev_decl, &versions, self.n_versions);
                        elements.push(element);
                        versions.clear();
                    }
                }
                versions.push(version);
                prev_decl = Some(decl.into());
            }
            if !versions.is_empty() {
                if let Some(decl) = prev_decl {
                    let element = LibItem::new(decl, &versions, self.n_versions);
                    elements.push(element);
                }
            }
        }

        let LibItems {
            adapter_functions,
            members,
            init_members,
            init_fields,
        } = elements.into();
        Ok(quote! {
            #(#adapter_functions)
            *

            pub struct Lib{
                __library: ::libloading::Library,
                #(#members),
                *
            }

            impl Lib{
                pub unsafe fn new<P>(path: P) -> Result<Self, ::libloading::Error>
                where
                    P: AsRef<::std::ffi::OsStr>,
                {
                    let library = ::libloading::Library::new(path)?;
                    Self::from_library(library)
                }
                pub unsafe fn from_library<L>(library: L) -> Result<Self, ::libloading::Error>
                where
                    L: Into<::libloading::Library>,
                {
                    let __library = library.into();
                    #(#init_members);
                    *
                    Ok(Self{
                        __library,
                        #(#init_fields), *
                    })
                }

            }

        })
    }
}

fn version_to_feature(version: &Version) -> String {
    format!(
        "cuda-{:0>2}{:0>2}{}",
        version.major, version.minor, version.patch
    )
}

pub fn merge<P: AsRef<Path>>(
    binding_dir: P,
    output_filename: P,
    lib_names: Vec<String>,
) -> Result<()> {
    let binding_dir = binding_dir.as_ref();
    let mut merger = BindingMerger::new(lib_names);

    let entries = fs::read_dir(binding_dir)?;
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
    let parsed = syn::parse2(unified.clone())
        .with_context(|| format!("In module {:?}", binding_dir.display()))?;
    std::fs::write(&output_filename, prettyplease::unparse(&parsed))?;
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

pub fn merge_bindings(modules: &[ModuleConfig]) -> Result<()> {
    for config in modules {
        merge(
            format!("out/{}/sys/linked", config.cudarc_name),
            format!("../src/{}/sys/mod.rs", config.cudarc_name),
            config.libs.clone(),
        )?;
    }
    Ok(())
}
