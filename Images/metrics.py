from sklearn.metrics import confusion_matrix

def accuracy_noavg(y, outs, **kwargs):
    # Reduction is not average, but sum here.
    pred = outs.max(dim=1)[1]
    ans = float(pred.eq(y.data).cpu().numpy().sum())
    return 100.0 * ans

def cm_noavg(y, outs, **kwargs):
    # Compute confusion matrix. cm[i, j] : True label i, Predicted label j
    pred = outs.max(dim=1)[1]
    cm = confusion_matrix(y.data.cpu().numpy(), pred.data.cpu().numpy())
    return cm

