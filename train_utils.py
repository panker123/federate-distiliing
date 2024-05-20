
import torch
import numpy as np
from transformers import T5ForConditionalGeneration
from transformers.trainer_utils import set_seed
from model_utils import TaskPrefixDataCollator
from torch.utils.data import  DataLoader

def train_model(args, run, tokenizer, tokenized_datasets,w_glob):
    # 设置随机种子
    set_seed(run)
    device='cuda'if torch.cuda.is_available()else"cpu"
    # 从预训练模型加载T5模型
    model=T5ForConditionalGeneration.from_pretrained("model")
    model.to(device)
    model.load_state_dict(w_glob)
    optim=torch.optim.Adam(model.parameters(),lr=5e-5)
    data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    train_dataloader=DataLoader(tokenized_datasets,batch_size=4,collate_fn=data_collator)
    total_loss=0
    n=0
    for data in train_dataloader:
        pred_data=data['pred']
        pred_data=pred_data.to(device)
        expel_data=data['expl']
        expel_data=expel_data.to(device)
        outputs1=model(**pred_data)
        outputs2=model(**expel_data)
        loss=outputs1.loss*0.5+outputs2.loss*0.5
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss+=loss
        n+=1

    changes = {k: model.state_dict()[k] - w_glob[k] for k in w_glob}
    print(total_loss / n)
    return changes

def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer


def eval_model(args, run, tokenizer, tokenized_datasets,w_glob):
    # 设置随机种子
    set_seed(run)
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    # 从预训练模型加载T5模型
    model_test = T5ForConditionalGeneration.from_pretrained("model")
    model_test.to(device)
    model_test.load_state_dict(w_glob)
    data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model_test)
    test_dataloader = DataLoader(tokenized_datasets, batch_size=1, collate_fn=data_collator)
    total_acc=0
    total_loss=0
    n=0
    print('------eval-------')
    for data in test_dataloader:
        pred_data = data['pred']
        pred_data = pred_data.to(device)
        outputs1 = model_test(**pred_data)
        loss = outputs1.loss
        total_loss+=loss.data
        #print(loss)
        pre=outputs1.logits.argmax(dim=-1)
        decoded_preds=tokenizer.batch_decode(pre,skip_special_tokens=True)
        labels=pred_data['labels']
        labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        # 使用 tokenizer 将真实标签解码成文本
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # 计算准确率
        if args.dataset !='svamp':
            acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
        else:
            preds=list()
            for pred in decoded_preds:
                preds.append(eval_equation(pred))
            lbls=list()
            for lbl in decoded_labels:
                lbls.append(eval_equation(lbl))
            acc=np.mean(np.array(preds) == np.array(lbls))
        total_acc+=acc
        n+=1
    del model_test
    print("eval acc:{}".format(total_acc/n))
    print("eval loss:{}".format(total_loss/n))
    return total_acc/n