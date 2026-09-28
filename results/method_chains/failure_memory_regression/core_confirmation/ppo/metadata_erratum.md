# PPO protocol metadata erratum

The frozen `protocol.json` records `version: nl-bidirectional-v1` because the
PPO bank was created before the protocol label was made SUT-specific. The
label is incorrect; the protocol's `builds` are `ppo_release_v0`,
`ppo_release_v1`, and `ppo_release_v2`, and its selector config records all
three PPO checkpoint hashes. This naming error does not change the frozen
manifest, physical runner, selector, or measured outcomes. New PPO protocol
freezes use `ppo-bidirectional-v1`.
