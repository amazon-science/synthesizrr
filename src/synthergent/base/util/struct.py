from typing import *
import math, gc, collections
from synthergent.base.util.language import MutableParameters, ProgressBar, is_list_like, as_list, set_param_from_alias
from pydantic import conint, confloat


class Trie(MutableParameters):
    parent: Optional['Trie'] = None
    value: Optional[Any] = None
    children: Dict[str, 'Trie'] = dict()
    _depth: Optional[conint(ge=0)] = None
    _max_child_depth: Optional[conint(ge=0)] = None
    _num_children_in_subtree: Optional[conint(ge=0)] = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        out: str = f'{self.class_name}('
        if self.value is not None:
            out += f'value={self.value}, '
        out += f'depth={self.depth}, num_children={self.num_children}'
        if self.has_children:
            out += f', children={set(self.children.keys())}'
        out += f')'
        return out

    @property
    def depth(self) -> int:
        """Calculates and returns depth of the current node. Root has depth of 0."""
        if self._depth is None:
            if self.parent is None:
                self._depth = 0
            else:
                self._depth = self.parent.depth + 1
        return self._depth

    @property
    def root(self) -> 'Trie':
        cur_node: Trie = self
        while self.parent is not None:
            cur_node: Trie = self.parent
        return cur_node

    @property
    def has_children(self) -> bool:
        return self.num_children > 0

    @property
    def num_children(self) -> int:
        return len(self.children)

    @property
    def num_nodes(self) -> int:
        return self.root.num_children_in_subtree

    @property
    def num_children_in_subtree(self) -> int:
        if self._num_children_in_subtree is None:
            if self.has_children is False:
                self._num_children_in_subtree = 0
            else:
                self._num_children_in_subtree: int = sum([
                    child.num_children_in_subtree for child in self.children.values()
                ]) + self.num_children
        return self._num_children_in_subtree

    @property
    def max_child_depth(self) -> int:
        if self._max_child_depth is None:
            if not self.has_children:
                self._max_child_depth: int = self.depth
            else:
                self._max_child_depth: int = max([child.max_child_depth for child in self.children.values()])
        return self._max_child_depth

    @property
    def max_depth(self) -> int:
        return self.root.max_child_depth

    def __getitem__(self, key):
        return self.children[key]

    @classmethod
    def of(
            cls,
            nodes: Union[List[List[str]], List[str], str],
            splitter: Optional[str] = None,
            allow_end_at_branch: bool = True,
            **kwargs,
    ) -> Any:
        """
        Creates a trie from a list of strings.
        Each node in the trie is a dict with further subdicts. Leafs are identified as dicts with '__end__' in them.
        Ref: https://stackoverflow.com/a/11016430
        """
        if isinstance(nodes, (str, set)):
            nodes: List[str] = as_list(nodes)

        assert is_list_like(nodes)
        set_param_from_alias(params=kwargs, param='progress_bar', alias=['progress', 'pbar'], default=True)
        pbar: ProgressBar = ProgressBar.of(
            kwargs.get('progress_bar'),
            miniters=1000,
            total=len(nodes),
            prefer_kwargs=True,
        )

        trie_root: Trie = Trie()
        try:
            for node_i, node in enumerate(nodes):
                if isinstance(node, str):
                    if splitter is None:
                        raise ValueError(f'When passing nodes as a list of strings, please pass `splitter`.')
                    node: List[str] = node.split(splitter)
                current_node: Trie = trie_root
                # print(f'\ncreating: {node}')
                for node_part_i, node_part in enumerate(node):
                    # print(f'\t{node_part}')
                    if node_part not in current_node.children:
                        ## For a path like 'A/B/C/D', create intermediate:
                        current_node.children[node_part] = Trie(parent=current_node)
                    current_node_child: Trie = current_node.children[node_part]
                    if allow_end_at_branch is False and node_part_i != len(node) - 1:
                        if len(current_node_child.children) == 0:
                            raise ValueError(
                                f'Branch nodes cannot be values for this Trie; thus cannot create trie from {node}'
                            )
                    current_node: Trie = current_node_child
                pbar.update(1)
                if node_i % 10_000 == 0:
                    gc.collect()
        finally:
            gc.collect()
        return trie_root
