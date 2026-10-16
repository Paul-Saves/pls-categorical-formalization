import Lake
open Lake DSL

package "pls_spd" where
  version := v!"0.1.0"

require "leanprover-community" / "mathlib" @ git
  "v4.14.0"

@[default_target]
lean_lib «FormalProofs» where
  roots := #[`FormalProofs]

lean_lib «AristotleExport» where
  roots := #[`AristotleExport]
