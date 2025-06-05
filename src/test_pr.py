import numpy as np
from utils.eval_utils import compute_precision_recall
from scipy.linalg import sqrtm

def calculate_fid(real_features, fake_features):
    """
    Calculate the FID score between two sets of features.
    This is a placeholder function; replace with actual FID calculation.
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

test_features = np.random.randn(10000, 128)

# # Test identical features
# precision, recall = compute_precision_recall(test_features, test_features.copy(), k=1)
# print(f"Identical features test: P={precision:.4f}, R={recall:.4f}")

# Test different features  
#different_features = np.random.randn(10000, 128)
#take half of the features from the test set and half from a far away distribution
#different_features = np.concatenate((test_features[:5000], np.random.randn(5000, 128) + 50), axis=0)
# generare different features by adding noise
different_features = test_features + np.random.randn(*test_features.shape) * 0.5
precision, recall = compute_precision_recall(test_features, different_features, k=3)
print(f"Different features test: P={precision:.4f}, R={recall:.4f}")

fid_score = calculate_fid(test_features, different_features)
print(f"FID score: {fid_score:.4f}")