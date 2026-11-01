"""Exact-input FP32 frozen-feature cache; caller checks a live reload parity."""
import hashlib
from pathlib import Path
import tempfile


class FeatureCache:
    def __init__(self,directory,encoder_sha256,runtime_version):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.encoder=encoder_sha256;self.runtime=runtime_version;self.hits=0;self.misses=0

    def metadata(self,tensor):
        if tensor.requires_grad:raise ValueError('Only frozen input features may be cached')
        return {'schema_version':1,'encoder_sha256':self.encoder,'runtime_version':self.runtime,
                'input_shape':list(tensor.shape),'input_dtype':str(tensor.dtype),
                'input_sha256':hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()}

    def path(self,metadata):
        import json
        key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
        return self.directory/(key+'.pt')

    @staticmethod
    def digest(features):
        value=hashlib.sha256()
        for feature in features:
            value.update(str((tuple(feature.shape),str(feature.dtype))).encode())
            value.update(feature.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    def get(self,tensor,compute):
        import torch
        metadata=self.metadata(tensor);path=self.path(metadata)
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if payload.get('metadata')!=metadata:raise ValueError('Feature-cache provenance mismatch')
            features=payload['features'];self.hits+=1
            if not isinstance(features,list) or len(features)!=3 or any(t.dtype!=torch.float32 or t.requires_grad or not torch.isfinite(t).all() for t in features):raise ValueError('Invalid cached FP32 features')
            if payload.get('features_sha256')!=self.digest(features):raise ValueError('Feature-cache content checksum mismatch')
            return features,True
        raw=compute()
        if any(x.requires_grad for x in raw):raise ValueError('Trainable features cannot use the frozen cache')
        features=[x.detach().cpu().contiguous().clone() for x in raw]
        if len(features)!=3 or any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in features):raise ValueError('Expected three finite FP32 feature maps')
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory,suffix='.tmp',delete=False) as handle:
                temporary=Path(handle.name)
                torch.save({'metadata':metadata,'features':features,'features_sha256':self.digest(features)},handle)
            temporary.replace(path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
        self.misses+=1
        return features,False
