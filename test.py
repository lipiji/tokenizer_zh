from gptx_tokenization import GPTXTokenizer

tknizer = GPTXTokenizer(model_file="gptx.model")

s = "我是中国人     我爱中国\n\n"
print(s)

print("tokenizing..")
tks = tknizer.tokenize(s)
print(tks)

print("encodng...")
ids = tknizer.encode(s)
print(ids)

print("decoding from tks")
print(tknizer.decode(tks))

print("decoding from ids")
print(tknizer.decode(ids))
