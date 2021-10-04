import time
from datasets.dataloader import emb_dataloader
from utils.evaluate import baseline_evaluate
from embedding.get_embedding import embedding
from pyod.models.ocsvm import OCSVM


def ocsvm(args,data):

			datadict=emb_dataloader(args)
			t0 = time.time()
			embeddings=embedding(args,datadict)

			dur1=time.time() - t0

			if args.mode=='A':
				data =  datadict['features']
				#print('A shape',data.shape)
			if args.mode=='X':
				data = embeddings
			clf = OCSVM(contamination=0.1)

			t1 = time.time()
			clf.fit(data[datadict['train_mask']])
			dur2=time.time() - t1

			print('traininig time:', dur1+dur2)

			t2 = time.time()
			y_pred_val=clf.predict(data[datadict['val_mask']])
			y_score_val=clf.decision_function(data[datadict['val_mask']])
			auc,ap,f1,acc,precision,recall=baseline_evaluate(datadict,y_pred_val,y_score_val,val=True)

			dur3=time.time() - t2
			print('infer time:', dur3)
			y_pred_test=clf.predict(data[datadict['test_mask']])
			y_score_test=clf.decision_function(data[datadict['test_mask']])
			auc,ap,f1,acc,precision,recall=baseline_evaluate(datadict,y_pred_test,y_score_test,val=False)


			return auc,ap,acc

