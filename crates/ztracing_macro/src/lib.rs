//! vue-native's Apache-2.0 replacement for zed's GPL `ztracing_macro`:
//! a pass-through `#[instrument]` that returns the item unchanged.

#[proc_macro_attribute]
pub fn instrument(
    _args: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    item
}
