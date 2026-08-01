Experimental selection protocol. For each seed, training and validation use a
persisted development split. The best checkpoint is selected using the equal
dataset-level mean of validation Dice on Kvasir-SEG and CVC-ClinicDB, frozen,
and only then evaluated on external test datasets. Test data do not influence
checkpoint, epoch, threshold, or configuration selection.
