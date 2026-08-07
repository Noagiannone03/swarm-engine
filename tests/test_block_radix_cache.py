from parallax.server.block_radix_cache import BlockRadixCache


def test_sibling_blocks_with_same_first_token_remain_reachable():
    cache = BlockRadixCache(block_size=2)
    root_branch = cache.insert_block([1, 2], block_id=10)

    cache.insert_block([7, 8], block_id=11, parent_path=[root_branch])
    cache.insert_block([7, 9], block_id=12, parent_path=[root_branch])

    assert cache.match_prefix([1, 2, 7, 8]) == ([10, 11], 4)
    assert cache.match_prefix([1, 2, 7, 9]) == ([10, 12], 4)
    assert cache.num_cached_blocks == 3


def test_evict_reclaims_all_colliding_sibling_blocks():
    freed_blocks = []
    cache = BlockRadixCache(block_size=2, on_block_evict=freed_blocks.append)
    root_branch = cache.insert_block([1, 2], block_id=10)

    cache.insert_block([7, 8], block_id=11, parent_path=[root_branch])
    cache.insert_block([7, 9], block_id=12, parent_path=[root_branch])

    assert cache.evict_lru_blocks(3) == 3
    assert set(freed_blocks) == {10, 11, 12}
    assert cache.num_cached_blocks == 0
    assert cache.root.children == {}


def test_existing_block_debug_log_does_not_expose_token_ids(caplog):
    cache = BlockRadixCache(block_size=2)
    token_ids = [987654321, 123456789]
    cache.insert_block(token_ids, block_id=10)

    with caplog.at_level("DEBUG"):
        cache.insert_block(token_ids, block_id=11)

    assert "987654321" not in caplog.text
    assert "123456789" not in caplog.text
    assert "token_count=2" in caplog.text
