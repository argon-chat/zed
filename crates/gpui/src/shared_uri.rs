use crate::util::impl_deref;

use crate::SharedString;

/// A [`SharedString`] containing a URI.
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct SharedUri(SharedString);

impl_deref!([] SharedUri => mut 0: SharedString);

impl std::fmt::Debug for SharedUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::fmt::Display for SharedUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_ref())
    }
}

impl<T: Into<SharedString>> From<T> for SharedUri {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}
