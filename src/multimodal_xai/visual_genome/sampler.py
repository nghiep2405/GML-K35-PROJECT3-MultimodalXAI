"""Seed-and-epoch deterministic, per-image rotation; no global RNG state."""
from __future__ import annotations
import hashlib
from collections import defaultdict
import numpy as np


def _cycle_permutation(values: tuple[str, ...], image_id: str, seed: int, cycle: int):
    digest=hashlib.sha256(f'gml-v2|{seed}|{image_id}|{cycle}'.encode()).digest()
    rng=np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    return [values[int(i)] for i in rng.permutation(len(values))]


class EpochQuerySampler:
    def __init__(self, rows, seed=2026, per_image=5):
        assert per_image >= 1
        groups=defaultdict(set)
        for row in rows:
            if isinstance(row, dict):
                image_id,text_id=str(row['positive_image_id']),str(row['text_id'])
            else:
                image_id,text_id=str(row[0]),str(row[1])
            groups[image_id].add(text_id)
        self.groups={image_id:tuple(sorted(ids)) for image_id,ids in sorted(groups.items())}
        self.seed=int(seed); self.per_image=int(per_image)
        if not self.groups: raise ValueError('Empty query pool')

    def sample_image(self, image_id, epoch):
        """Epoch is zero-indexed; same seed+pool+epoch gives same exact IDs."""
        if epoch < 0: raise ValueError('epoch must be >=0')
        values=self.groups[str(image_id)]; n=len(values)
        count=min(self.per_image,n)
        # Full n=20: four nonoverlapping batches of 5 per cycle.
        start=int(epoch)*count
        result=[]; selected=set(); position=start
        # Independent per-image and per-cycle permutations; no reliance on input ordering.
        while len(result)<count:
            cycle,idx=divmod(position,n)
            perm=_cycle_permutation(values,str(image_id),self.seed,cycle)
            candidate=perm[idx];position+=1
            if candidate not in selected:
                result.append(candidate);selected.add(candidate)
        return result

    def sample(self, epoch):
        return [(image_id,tid) for image_id in self.groups for tid in self.sample_image(image_id,epoch)]

    def coverage(self, epochs):
        observed={image_id:set() for image_id in self.groups}
        by_epoch=[]
        for epoch in range(epochs):
            sample=self.sample(epoch)
            for image_id,tid in sample:observed[image_id].add(tid)
            by_epoch.append({'epoch':epoch+1,'queries':len(sample),'coverage':sum(map(len,observed.values()))/sum(map(len,self.groups.values()))})
        return by_epoch
