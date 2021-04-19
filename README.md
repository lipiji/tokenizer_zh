# tokenizer_zh
- Jieba + BPE

- Trained based on 10G Chinese corpus (news, wiki, poetry).

- Example
```
tknizer = GPTXTokenizer(model_file="gptx.model")

s = "我是中国人     我爱中国\n\n"

tks = tknizer.tokenize(s)
print(tks)
->['▁我', '▁是', '▁中国', '▁人', '▁<space>', '▁<space>', '▁<space>', '▁<space>', '▁<space>', '▁我', '▁爱', '▁中国', '▁<eol>', '▁<eol>']

ids = tknizer.encode(s)
print(ids)
->[43, 30, 158, 35, 3, 3, 3, 3, 3, 43, 263, 158, 4, 4]

print(tknizer.decode(tks))
->我是中国人     我爱中国



print(tknizer.decode(ids))
->我是中国人     我爱中国


```
