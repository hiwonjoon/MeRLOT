from .multi_dense import MultiInputDense
MultiInputMultiDense = MultiInputDense # Old name

from .attention import MultiHeadedAttention, Encoder, Decoder, dot_product_attention

from .prediction import ProbabilisticPrediction, Prediction
ProbabilisticFinalPrediction = ProbabilisticPrediction # Old Name
FinalPrediction = Prediction # Old name

from .updater import MetaFunUpdater, MetaFunUpdaterLocal
