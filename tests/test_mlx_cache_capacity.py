from parallax.server.cache_manager import cap_blocks_to_token_ceiling


def test_cache_blocks_are_capped_to_the_servable_token_window():
    assert cap_blocks_to_token_ceiling(30_265, 32, 65_536) == 2_048


def test_cache_block_cap_rounds_up_the_last_partial_block():
    assert cap_blocks_to_token_ceiling(100, 32, 33) == 2


def test_cache_block_cap_preserves_a_smaller_memory_bound_pool():
    assert cap_blocks_to_token_ceiling(64, 32, 65_536) == 64
    assert cap_blocks_to_token_ceiling(64, 32, None) == 64
