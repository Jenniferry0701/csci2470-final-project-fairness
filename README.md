# csci2470-final-project-fairness

### what yujia added
1. adult census and compas data
2. a class to evaluate model performance and fairness metrics
3. a vanilla DNN

### what vivian added
1. default dataset
2. data processing utils file
    - can modify the columns which appear in the processed dataset
    - TODO: is there a set rule for converting non-numerical attributes 
    (in particular for protected attributes i'm assuming we should have 'male' -> 1, 'female' -> 0 if we are considering 0 to be "unprivileged" and 1 to be "privileged")
3. refactored init.py (specify dataset in command line args)

### what yujia added
1. added nationality as a potential protected field in adult dataset
2. added saving results to the output file
3. added intersectional fairness metrics that capture multiple groups 

### TODO
* refactor code
    - determine how to set the "privileged" vs "unprivileged" values depending on which protected attributes are being considered
    - (see get_group_metrics in metrics.py)
* make it easier to switch between datasets
    - currently can be done via command line args
* seems more complicated to include mep15/16 datasets - can maybe stick with datasets we already have
* apply adverserial learning and compare with vanilla network
* implement intersectional fairness metrics - taking account of multiple identities when calculating metrics
