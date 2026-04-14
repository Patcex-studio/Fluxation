# Adaptiflux Core

`adaptiflux-core` is a Rust library implementing the core architecture for the Adaptiflux distributed self-organizing swarm framework.

## Overview

Adaptiflux Core provides modular systems for agents, rules, primitives, topology management, and message delivery abstractions. The library is designed for extensibility, adaptive execution, and integration with optional optimization backends.

The framework implements hybrid computational models combining:
- **Hybrid Logics**: Neural-inspired plasticity with evolutionary optimization
- **Self-Organization**: Dynamic topology adaptation through plasticity rules
- **Plasticity**: Synaptogenesis, pruning, apoptosis, and neurogenesis mechanisms
- **Learning**: Online adaptation engines with gradient descent and evolutionary learners
- **Attention and Memory**: Content-based attention and long-term memory storage
- **Hierarchical Scaling**: Abstraction layers for managing large-scale agent swarms

## Installation

Add `adaptiflux-core` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
adaptiflux-core = "0.1.0"
```

For optional features:

```toml
[dependencies]
adaptiflux-core = { version = "0.1.0", features = ["gpu", "adaptiflux_optim"] }
```

## Quick Start

Here's a simple example of creating and running a basic scheduler with an agent:

```rust,no_run
use adaptiflux_core::{CoreScheduler, AgentBlueprint, ZoooidHandle, MessageBus};

struct SimpleAgent;

impl AgentBlueprint for SimpleAgent {
    fn update(&mut self, _messages: Vec<adaptiflux_core::Message>) -> adaptiflux_core::AgentUpdateResult {
        // Your agent logic here
        adaptiflux_core::AgentUpdateResult::NoChange
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a scheduler
    let mut scheduler = CoreScheduler::new();

    // Add an agent
    let agent = Box::new(SimpleAgent);
    let _handle = scheduler.add_agent(agent)?;

    // Run the scheduler
    scheduler.run()?;

    Ok(())
}
```

For more examples, see the `examples/` directory.

## Documentation

- **API Documentation**: Run `cargo doc --open` to view the full API documentation locally.
- **Online Docs**: Available on [docs.rs](https://docs.rs/adaptiflux-core) (once published).
- **Architecture**: See `docs/architecture.md` for detailed architectural overview.
- **Rules and Consistency**: See `docs/rules_and_consistency.md` for plasticity rules documentation.

## Repository layout

- `src/` — library source code.
- `docs/` — architecture and API documentation.
- `examples/` — sample applications and experiment runners.
- `benches/` — benchmark harnesses.
- `tests/` — integration test scenarios.
- `archive/` — archived internal reports and intermediate experiment outputs.

## Requirements

- Rust stable toolchain
- `cargo`
- Optional: GPU toolchain for `custom_optim` CUDA support

## Build

```bash
cargo build
cargo build --features ui
cargo build --features adaptiflux_optim
cargo build --features custom_optim
```

## Test

```bash
cargo test
```

## Examples

Run the MNIST learning example:

```bash
cargo run --example mnist_learning
```

Run the full experiment example:

```bash
cargo run --example mnist_full_experiment
```

Use optional features:

```bash
cargo build --features ui
cargo build --features adaptiflux_optim
cargo build --features custom_optim
```

## Documentation

- Architecture overview: `docs/architecture.md`
- Rules and consistency: `docs/rules_and_consistency.md`
- Archived internal reports and experiment outputs: `archive/`

## Contributing

See `CONTRIBUTING.md` for contribution guidelines, PR workflow, and test instructions.

## License

<!--
Copyright (C) 2026 Jocer S. <patcex@proton.me>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

SPDX-License-Identifier: AGPL-3.0 OR Commercial
-->

This project is dual-licensed:

- **GNU AGPLv3**: For open-source and community use. See [LICENSE](LICENSE).
- **Commercial License**: For proprietary, closed-source, or non-AGPL use. See [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) for details and contact information.

By default, you may use this project under the terms of the GNU Affero General Public License v3.0 or later (AGPLv3). If you wish to use the code in a proprietary product or do not wish to comply with AGPLv3, you must obtain a commercial license.

For commercial licensing, please contact:

- Email: patcex@proton.me
- Author: https://github.com/Jocer-Speis

SPDX-License-Identifier: AGPL-3.0 OR Commercial