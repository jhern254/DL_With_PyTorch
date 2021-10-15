import copy
import csv
import functools
import glob
import os

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from logconf import logging         # util.logconf breaks, don't know why
from util.util1 import XyzTuple, xyz2irc

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw') # have to define

# clean tuple data structure for raw data
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',                                   # value 1
    'isNodule_bool, diameter_mm, series_uid, center_xyz'    # value 2
)

# in memory caching decorator
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):  # only focuses on disk data
    '''
        We construct a set with all series_uids that are present on disk.
        This will let us use the data even if we haven't downloaded all of 
        the subsets yet.
    '''
    # find all files, glob returns list
    mhd_list = glob.glob('../data/subset*/*.mhd')
    # turns into set, only keeps uid
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # make dict from annotations.csv
    diameter_dict = {}
    # open file for reading
    with open('../data/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                    (annotationCenter_xyz, annotationDiameter_mm)
            )

    # make list from candidates.csv
    candidateInfo_list = []
    with open('../data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # skip if uid not in disk
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            # check data so diameter size is same for both sets,if not set to 0
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    # check if distance is too far to be same nodule
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                    else:
                        candidateDiameter_mm = annotationDiameter_mm
                        break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz
            ))

    # sort data large to small nodule
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid):
        # finds path of uid from all subsets
        mhd_path = glob.glob(
            '../data/subset*/{}.mhd'.format(series_uid)
        )[0]

        # use sitk to convert .raw/.mhd catscan image to tensor (1, 123, 512, 512)
        ct_mhd = sitk.ReadImage(mhd_path) # 3 spatial dims (C x Depth x W x H)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

# CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
# HU are scaled oddly, with 0 g/cc (air, approximately) being 
# -1000 and 1 g/cc (water) being 0.
# The lower bound gets rid of negative density stuff used to indicate out-of-FOV
# The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        # make inputs for xyz2irc coordinate conversion
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        # converts direction to array, reshape array to 3x3
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        # returns center in irc coords
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )

        slice_list = []
        # creates CT slice based on index
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], (repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis]))

            if start_ndx < 0:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                    self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                    self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        # returns cubic CT chunk , irc center
        return ct_chunk, center_irc

# on disk caching functions
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    """ Caches Ct instance for on disk repeated use """
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc
                

class LunaDataset(Dataset):
    '''
        Input Ct instance(hundreds of samples), normalize,flatten into 
        single collection from which samples can be terieved without
        regard for which Ct instance the sample originates.
        Any subclass of Dataset needs __len__ and __getitem__
        val_stride: sets validation set from data, for every n
        isValSet_bool: if validation set
    '''
    def __init__(self, 
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        # sets nodules from passed in series_uid
        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        
        # splits training/validation set
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            # delete valid. set index from training set
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        
        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training"
        ))
        

    def __len__(self):
        return len(self.candidateInfo_list) # size N

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        # cand_a is (32, 48, 48) (Depth x W x H)
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc
        )

        # Get data into proper type/dims.
        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0) # add batch dim. B

        # classification tensor - nodule/not nodule, pos./neg. result
        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long
        )

        # training sample, returning 2 tensors
        return(candidate_t,         # size N-1
                pos_t,
                candidateInfo_tup.series_uid,
                torch.tensor(center_irc)        # needed input tensor
        )




if __name__ == "__main__":
#    list0 = glob.glob('../data/subset0/{}.mhd'.format(series_uid))[0]
#    print(list0)


#    mhd_list = glob.glob('../data/subset0/*.mhd')

    # Testing fns
    if(True):
        x = getCandidateInfoList()
        print(x[0])
        print('###\n')

        y = Ct(x[0][2])
        print(y.direction_a)
        a, b = y.getRawCandidate(y.origin_xyz, y.vxSize_xyz)
        print('\nTest main\n')
        print(a)
        print(b)
    
#    tens = torch.randn(32, 48, 48)
#    tens_new = tens.unsqueeze(0)
#    print(tens_new.shape)

    if(True):
        candidateInfo_tup = x[0]
        width_irc = (32, 48, 48)

        # cand_a is (32, 48, 48) (Depth x W x H)
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc
        )
        print(candidate_a.shape, candidate_a.dtype)
        print(center_irc)

    print('\n\nEnd main')




