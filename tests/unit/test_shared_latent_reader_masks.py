from types import SimpleNamespace
import numpy as np
import pytest
from scripts.hpc.probe_shared_latent_reader_masks import raw_mask_statistics


def fixture():
    schema=SimpleNamespace(fields=(SimpleNamespace(name='field', shape=(2,3)),))
    raw={'field':np.array([[2,0,3],[0,4,0]],np.float32)}
    masks={'field':np.array([[True,False,True],[False,True,False]])}
    return raw,masks,schema


def test_actual_mask_locations_have_exact_zero():
    result=raw_mask_statistics(*fixture())
    assert result['inactive_coordinates']==3
    assert result['fields']['field']['inactive_nonzero']==0


def test_even_tiny_inactive_nonzero_is_rejected():
    raw,masks,schema=fixture()
    raw['field'][1,0]=np.float32(1e-30)
    with pytest.raises(ValueError,match='nonzero'):
        raw_mask_statistics(raw,masks,schema)


def test_nonfinite_inactive_and_missing_output_rejected():
    raw,masks,schema=fixture(); raw['field'][0,1]=np.nan
    with pytest.raises(ValueError,match='finite'):
        raw_mask_statistics(raw,masks,schema)
    with pytest.raises(ValueError,match='field sets'):
        raw_mask_statistics({},masks,schema)


def test_fixture_without_inactive_coordinates_cannot_certify_zeroing():
    raw,masks,schema=fixture(); masks['field'][:]=True
    with pytest.raises(ValueError,match='no inactive'):
        raw_mask_statistics(raw,masks,schema)
