Shared training objective. Every variant uses the same weighted combination of
soft Dice, label-smoothed binary cross-entropy, and a boundary term formed by
weighted binary cross-entropy plus weighted IoU loss. When auxiliary outputs are
present, the fixed deep-supervision weights are applied in output order.
