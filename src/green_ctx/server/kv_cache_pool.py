from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, List
import logging


SANITY_CHECK = False
PAGE_BYTES = 2 * 1024 * 1024  # 2MB

logger = logging.getLogger(__name__)


class Page:

    def __init__(self, page_id: int, page_bytes: int):
        self.page_id = page_id
        self.page_bytes = page_bytes
        self.num_kv_blocks = None
        self.kv_block_bytes = None

        self.stt_block_id = None
        self.end_block_id = None
        self.free_list = None

    def init(self, kv_block_bytes: int) -> None:
        assert not self.inited()

        self.stt_block_id = self.page_id * self.page_bytes // kv_block_bytes
        self.end_block_id = (self.page_id +
                             1) * self.page_bytes // kv_block_bytes
        if self.page_bytes % kv_block_bytes != 0:
            # Skip the first block which is not aligned.
            self.stt_block_id += 1

        self.kv_block_bytes = kv_block_bytes
        self.num_kv_blocks = self.end_block_id - self.stt_block_id
        self.free_list = list(range(self.stt_block_id, self.end_block_id))

    def reset(self) -> None:
        assert self.inited() and len(self.free_list) == self.num_kv_blocks
        self.num_kv_blocks = None
        self.kv_block_bytes = None
        self.stt_block_id = None
        self.end_block_id = None
        self.free_list = None

    def inited(self) -> bool:
        return self.num_kv_blocks is not None and self.free_list is not None

    def alloc(self) -> int:
        if self.full():
            raise ValueError(f"Page {self.page_id} is already full")
        block_id = self.free_list.pop()
        return block_id

    def alloc_batch(self, num_blocks: int) -> List[int]:
        num_blocks = min(num_blocks, len(self.free_list))
        block_ids = self.free_list[:num_blocks]
        self.free_list = self.free_list[num_blocks:]
        return block_ids

    def free(self, block_id: int) -> None:
        if SANITY_CHECK:
            self._sanity_check(block_id)
        self.free_list.append(block_id)

    def free_batch(self, block_ids: List[int]) -> None:
        if SANITY_CHECK:
            for block_id in block_ids:
                self._sanity_check(block_id)
        self.free_list.extend(block_ids)

    def empty(self) -> bool:
        return len(self.free_list) == self.num_kv_blocks

    def full(self) -> bool:
        return not self.free_list

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def get_free_blocks(self) -> List[int]:
        return self.free_list

    def get_used_blocks(self) -> List[int]:
        all_blk_ids = [
            bid for bid in range(self.stt_block_id, self.end_block_id)
        ]
        return list(set(all_blk_ids) - set(self.free_list))

    def _has_block(self, block_id: int) -> bool:
        return block_id >= self.stt_block_id and block_id < self.end_block_id

    def _sanity_check(self, block_id: int) -> None:
        if not self._has_block(block_id):
            raise ValueError(
                f"Page {self.page_id} does not have block {block_id}")
        if block_id in self.free_list:
            raise ValueError(f"Block {block_id} is already free")


class PageAllocatorBase(ABC):

    @abstractmethod
    def __init__(self, total_mem_size: int, page_bytes: int):
        pass

    @abstractmethod
    def alloc_page(self) -> int:
        pass

    @abstractmethod
    def free_page(self, page: int) -> None:
        pass

    @abstractmethod
    def get_num_free_pages(self) -> int:
        pass

    @abstractmethod
    def get_num_total_pages(self) -> int:
        pass


class PageAllocator(PageAllocatorBase):

    def __init__(self,
                 total_mem_size: int,
                 page_bytes: int = PAGE_BYTES) -> None:
        logger.info(f"Init PageAllocator: "
                    f"total_mem_size={total_mem_size//(1024*1024)}MB, "
                    f"page_bytes={page_bytes//(1024*1024)}MB")
        self.total_mem_size = total_mem_size
        self.page_bytes = page_bytes
        self.num_total_pages = total_mem_size // page_bytes
        self.num_free_pages = total_mem_size // page_bytes
        self.free_page_list: deque[int] = deque(range(self.num_free_pages))

    def alloc_page(self) -> Page:
        if self.num_free_pages <= 0:
            raise ValueError("No free pages left")
        self.num_free_pages -= 1

        page_id = self.free_page_list.popleft()
        page = Page(page_id, self.page_bytes)

        return page

    def free_page(self, page: Page) -> None:
        page_id = page.page_id
        if SANITY_CHECK:
            if page_id in self.free_page_list:
                raise ValueError(f"Page {page_id} is already free")

        self.num_free_pages += 1
        self.free_page_list.append(page_id)

    def free_pages(self, page_ids: List[int]) -> None:
        self.num_free_pages += len(page_ids)
        if SANITY_CHECK:
            for page_id in page_ids:
                if page_id in self.free_page_list:
                    raise ValueError(f"Page {page_id} is already free")
        self.free_page_list.extend(page_ids)

    def get_num_free_pages(self) -> int:
        return self.num_free_pages

    def get_num_inuse_pages(self) -> int:
        return self.num_total_pages - self.num_free_pages

    def get_num_total_pages(self) -> int:
        return self.num_total_pages

    def get_page_id(self, block_id: int, kv_block_bytes: int) -> int:
        return block_id * kv_block_bytes // self.page_bytes

    def get_num_free_blocks(self, kv_block_bytes: int) -> int:
        return self.get_num_free_pages() * self._num_blocks_per_page(
            kv_block_bytes)

    def get_num_inuse_blocks(self, kv_block_bytes: int) -> int:
        return self.get_num_inuse_pages() * self._num_blocks_per_page(
            kv_block_bytes)

    def get_num_total_blocks(self, kv_block_bytes: int) -> int:
        return self.get_num_total_pages() * self._num_blocks_per_page(
            kv_block_bytes)

    def _num_blocks_per_page(self, kv_block_bytes: int):
        # When page_bytes is not aligned with kv_block_bytes, this function will
        # become inaccurate. Assuming page_bytes is always aligned here.
        assert self.page_bytes % kv_block_bytes == 0
        return self.page_bytes // kv_block_bytes


class KVCachePool:

    def __init__(self, num_blocks: int, kv_block_bytes: int) -> None:
        self.num_blocks = num_blocks
        self.kv_block_bytes = kv_block_bytes

        mem_size = self.num_blocks * self.kv_block_bytes
        self.page_allocator = PageAllocator(mem_size, PAGE_BYTES)

        self.num_avail_blocks = 0  # Only count free blocks in avail_pages
        self.avail_pages: Dict[int, Page] = {}
        self.full_pages: Dict[int, Page] = {}

        logger.info(
            f"Init KVCachePool: "
            f"num_blocks={num_blocks}, kv_block_bytes={kv_block_bytes}")
        if PAGE_BYTES % kv_block_bytes != 0:
            logger.warning(
                f"PAGE_BYTES={PAGE_BYTES} is not aligned with "
                f"kv_block_bytes={kv_block_bytes}")

    def alloc(self, need_size: int) -> List[int]:
        if self.available_size() < need_size:
            return None

        ret_idxes = []
        page: Page = None
        remaining_need = need_size
        while remaining_need > 0:
            if not self.avail_pages:
                page = self.page_allocator.alloc_page()
                page.init(self.kv_block_bytes)
                self.num_avail_blocks += page.num_free_blocks()
            else:
                _, page = self.avail_pages.popitem()
            assert page is not None
            if page.num_free_blocks() > remaining_need:
                self.num_avail_blocks -= remaining_need
                alloced_index = page.free_list[:remaining_need]
                page.free_list = page.free_list[remaining_need:]
                ret_idxes.extend(alloced_index)
                remaining_need = 0
                self.avail_pages[page.page_id] = page
            else:
                self.num_avail_blocks -= page.num_free_blocks()
                ret_idxes.extend(page.free_list)
                remaining_need -= len(page.free_list)
                page.free_list = []
                self.full_pages[page.page_id] = page
        assert remaining_need == 0, "Insufficient memory for allocation."
        return ret_idxes

    def free(self, indices: List[int]):
        idx_dict = defaultdict(list)
        for idx in indices:
            page_id = self.page_allocator.get_page_id(idx, self.kv_block_bytes)
            idx_dict[page_id].append(idx)

        pages_to_free: List[int] = []
        for page_id, idxs in idx_dict.items():
            if page_id in self.full_pages:
                page = self.full_pages.pop(page_id)
            else:
                page = self.avail_pages.pop(page_id)

            self.num_avail_blocks += len(idxs)
            page.free_batch(idxs)
            if page.empty():
                pages_to_free.append(page.page_id)
                self.num_avail_blocks -= page.num_free_blocks()
            else:
                self.avail_pages[page_id] = page

        self.page_allocator.free_pages(pages_to_free)

    def available_size(self) -> int:
        avail_size = self.num_avail_blocks
        free_size = self.page_allocator.get_num_free_blocks(
            self.kv_block_bytes)
        return avail_size + free_size

    def clear(self):
        raise NotImplementedError
