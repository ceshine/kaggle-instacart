import gc
import sys

# blend_gbm must be imported first.
# Otherwise will raise SIGSEGV when calling bst.train().
import basket.models.blend.gbm as blend_gbm
import basket.models.rnn_product.model as rnn_product_model
import basket.models.rnn_product.bmm_model as rnn_product_bmm_model
import basket.models.rnn_product.extract_states as rnn_product_extract
import basket.models.blend.optimization as opt
import basket.models.blend.merge as merge

if __name__ == "__main__":
    if len(sys.argv) == 2:
        SEED = int(sys.argv[1])
    else:
        SEED = 8898
    print("Using SEED", SEED)
    print("Training RNN PRODUCT BMM model...")
    rnn_product_bmm_model.main(SEED)
    gc.collect()
    print("Extracting states from RNN PRODUCT BMM model...")
    rnn_product_extract.main(bmm=True)
    print("Training RNN PRODUCT model...")
    rnn_product_model.main(SEED)
    gc.collect()
    print("Extracting states from RNN PRODUCT model...")
    rnn_product_extract.main(bmm=False)
    gc.collect()
    print("Merging Model Outputs...")
    merge.main()
    print("Training GBM meta model...")
    blend_gbm.main(SEED)
    gc.collect()
    print("Optimizing and generating submission...")
    opt.main()
