import multiprocessing
import regex as re
import numpy as np
import os
from typing import *
from collections import *
import json
from tqdm import tqdm
import time

#A BPE tokenizer implementation based on the paper "Neural Machine Translation of Rare Words with Subword Units" (https://arxiv.org/abs/1508.07909).
#For the CS190C class project only.
class Tokenizer:
    def __init__(self,vocabsize=10000,special_tokens:list[bytes]|None=[b"<|endoftext|>"],pat=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
        self.vocab:dict[int,bytes]={}
        self.inverse_vocab:dict[bytes,int]={}
        self.bpe_ranks:dict[tuple[bytes,bytes],int]={}
        self.special_token_ids:dict[bytes,int]={}
        self.merges:list[tuple[bytes,bytes]]=[]
        self.special_tokens=special_tokens
        self.pat=pat
        self.vocabsize=vocabsize

    def setvocabsize(self,vocabsize:int):
        self.vocabsize=vocabsize

    def setspecialtokens(self,special_tokens:list[bytes]):
        self.special_tokens=special_tokens

    def setpattern(self,pat):
        self.pat=pat

    def reset(self):
        self.vocab={}
        self.merges=[]
        self.inverse_vocab={}
        self.bpe_ranks={}
        self.special_token_ids={}

    #save all states in the self to the disk,remember the bytes can't direct for json
    def save(self,path:str):
        def bytes_to_int_list(data: bytes) -> list[int]:
            return list(data)

        state = {
            "vocab": [[token_id, bytes_to_int_list(token_bytes)] for token_id, token_bytes in self.vocab.items()],
            "inverse_vocab": [[bytes_to_int_list(token_bytes), token_id] for token_bytes, token_id in self.inverse_vocab.items()],
            "bpe_ranks": [
                [bytes_to_int_list(pair[0]), bytes_to_int_list(pair[1]), rank]
                for pair, rank in self.bpe_ranks.items()
            ],
            "special_token_ids": [
                [bytes_to_int_list(token_bytes), token_id]
                for token_bytes, token_id in self.special_token_ids.items()
            ],
            "merges": [
                [bytes_to_int_list(pair[0]), bytes_to_int_list(pair[1])]
                for pair in self.merges
            ],
            "special_tokens": (
                None
                if self.special_tokens is None
                else [bytes_to_int_list(token) for token in self.special_tokens]
            ),
            "pat": self.pat,
            "vocabsize": self.vocabsize,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    #load the Tokenizer from the disk
    @classmethod
    def load(cls,path:str):
        def int_list_to_bytes(data: list[int]) -> bytes:
            return bytes(data)

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        special_tokens_data = state.get("special_tokens")
        special_tokens = (
            None
            if special_tokens_data is None
            else [int_list_to_bytes(token) for token in special_tokens_data]
        )

        tokenizer = cls(
            vocabsize=state["vocabsize"],
            special_tokens=special_tokens,
            pat=state["pat"],
        )

        tokenizer.vocab = {
            int(token_id): int_list_to_bytes(token_bytes)
            for token_id, token_bytes in state["vocab"]
        }
        tokenizer.inverse_vocab = {
            int_list_to_bytes(token_bytes): int(token_id)
            for token_bytes, token_id in state["inverse_vocab"]
        }
        tokenizer.bpe_ranks = {
            (int_list_to_bytes(left), int_list_to_bytes(right)): int(rank)
            for left, right, rank in state["bpe_ranks"]
        }
        tokenizer.special_token_ids = {
            int_list_to_bytes(token_bytes): int(token_id)
            for token_bytes, token_id in state["special_token_ids"]
        }
        tokenizer.merges = [
            (int_list_to_bytes(left), int_list_to_bytes(right))
            for left, right in state["merges"]
        ]

        return tokenizer
        

    
    def find_chunk_boundaries( self,file: BinaryIO,desired_num_chunks: int,split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))



    #whole pretoken process
    def pretokenprocess(self,textfile:str|os.PathLike):
        with open(textfile, "rb") as f:
            num_processes = multiprocessing.cpu_count()*2#change for  your cpu cores
            #get bounaries
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            tasks=[(textfile,boundaries[i],boundaries[i+1]) for i in range(len(boundaries)-1)]
            counts=Counter()
            #get results from chunks
            with multiprocessing.Pool() as pool:
                results=pool.starmap(self.worker_task,tasks)
            for local_counts in results:
                counts.update(local_counts)
            return counts
        
    #split from special tokens    
    def splitspecialtokens(self,chunk):
        special_tokens=self.special_tokens
        if not special_tokens:
            return [chunk]
        # Match longer special tokens first to handle overlapping tokens correctly.
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        #escape special tokens to normal tokens
        escaped_tokens=[re.escape(tok.decode("utf-8")) for tok in special_tokens]
        split_pattern="|".join(escaped_tokens)
        #split
        textlist=re.split(split_pattern,chunk.decode("utf-8"))
        #remove empty
        return [texts for texts in textlist if texts]
    
    #process one chunk
    def chunkprocess(self,chunk):
        regexpattern=self.pat
        #split
        chunk_list=self.splitspecialtokens(chunk)
        local_counts=Counter()
        #use regex to match
        for chunk in chunk_list:
            for match in re.finditer(regexpattern,chunk):
                pre_token=match.group()
                bytesencoded=pre_token.encode("utf-8")
                #encode to bytes
                bytetuple=tuple(bytes([b]) for b in bytesencoded)
                local_counts[bytetuple]+=1

        return local_counts
    
    #worker function for multiprocessing,open file independently
    def worker_task(self,textfile:str,start:int,end:int):
        with open(textfile,"rb") as f:
            f.seek(start)
            chunk=f.read(end-start)
        local_counts=self.chunkprocess(chunk)
        return local_counts


    #replace the neighbour paired elements into one
    def replace_pair(self,word:tuple[bytes],pair:tuple[bytes,bytes]):
        new_word=[]
        i=0
        while i<len(word):
            if i<len(word)-1 and word[i]==pair[0] and word[i+1]==pair[1]:
                new_word.append(pair[0]+pair[1])
                i+=2
            else:
                new_word.append(word[i])
                i+=1
        return tuple(new_word)


    #main traning bpe function
    def train_bpe(self,textfile:str|os.PathLike,update=True):
        #get pretoken
        #we profile time here to make sure the pretoken process is efficient enough
        start_time = time.time()
        pretokencounts=self.pretokenprocess(textfile)
        pretokenendtime=time.time()
        print(f"Pretokenization took {pretokenendtime - start_time:.2f} seconds")
        #now profile the merge time to make sure it's efficient enough
        merge_start_time = time.time()
        vocabsize=self.vocabsize
        sl=0
        if self.special_tokens:
            sl=len(list(self.special_tokens))
        num_merges=vocabsize-256-sl
        #cache init
        paircounts:dict[tuple[bytes,bytes],int]={}
        pairtowords:dict[tuple[bytes,bytes],set[tuple[bytes]]]={}
        for word,count in pretokencounts.items():
            for i in range(len(word)-1):
                pair=(word[i],word[i+1])
                if pair not in paircounts.keys():
                    paircounts[pair]=0
                if pair not in pairtowords.keys():
                    pairtowords[pair]=set()
                paircounts[pair]+=count
                pairtowords[pair].add(word)
        initfinishtime=time.time()
        print(f"Initialization for merges took {initfinishtime - merge_start_time:.2f} seconds")
        #now start the loop
        merges:list[tuple[bytes, bytes]]=[]
        for _ in tqdm(range(num_merges)):
            if not paircounts:
                break
            #need to add p in key for the same count
            bestpair=max(paircounts.keys(),key=lambda p:(paircounts[p],p))
            #record it
            merges.append(bestpair)
            #update needed words
            wordneed=list(pairtowords[bestpair])
            for word in wordneed:
                count=pretokencounts[word]
                #remove the formal pair counts
                for i in range(len(word)-1):
                    p=(word[i],word[i+1])
                    if p not in paircounts.keys():
                        paircounts[p]=0
                    paircounts[p]-=count
                    if paircounts[p]<=0:
                        del paircounts[p]
                #update new word
                newword=self.replace_pair(word,bestpair)
                #update the wordcount
                del pretokencounts[word]
                pretokencounts[newword]=count
                #add the new pair counts
                for i in range(len(newword)-1):
                    p=(newword[i],newword[i+1])
                    if p not in paircounts:
                        paircounts[p]=0
                    if p not in pairtowords:
                        pairtowords[p]=set()
                    paircounts[p]+=count
                    pairtowords[p].add(newword)
        #self.merges=merges
        vocab:dict[int,bytes]={}
        current=0
        mergeendtime=time.time()
        print(f"Merge loop took {mergeendtime - initfinishtime:.2f} seconds")
        #now create vocab
        #basic 256
        for b in range(256):
            vocab[current]=bytes([b])
            current+=1
        #special tokens
        specialtokens=self.special_tokens
        if specialtokens:
            for token in specialtokens:
                vocab[current]=token
                current+=1
        #add new pairs
        for pair in merges:
            merged_pairs=pair[0]+pair[1]
            vocab[current]=merged_pairs
            current+=1
        #self.vocab=vocab
        if(update):
            self.vocab=vocab
            self.merges=merges
        finaltime=time.time()
        print(f"Creating vocab took {finaltime - mergeendtime:.2f} seconds")
        print(f"Total time for BPE training: {finaltime - start_time:.2f} seconds")
        return vocab,merges
    
    #build the tokenizer from training from file
    def build_from_file(self,textfile):
        #we add profile time here to make sure the training process is efficient enough
        start_time = time.time()
        self.train_bpe(textfile=textfile,update=True)
        trainendtime=time.time()
        print(f"Training BPE took {trainendtime - start_time:.2f} seconds")
        for key,value in self.vocab.items():
            self.inverse_vocab[value]=key
        for i in range(len(self.merges)):
            self.bpe_ranks[self.merges[i]]=i
        if self.special_tokens is not None:
            for tok in self.special_tokens:
                self.special_token_ids[tok]=self.inverse_vocab[tok]
        finalendtime=time.time()
        print(f"Building vocab and merges took {finalendtime - trainendtime:.2f} seconds")
        print(f"Building tokenizer from file took {finalendtime - start_time:.2f} seconds")
    
    #build from built vocabs
    def build_from_vocab_merges(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[bytes]|None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        for key,value in self.vocab.items():
            self.inverse_vocab[value]=key
        for i in range(len(self.merges)):
            self.bpe_ranks[self.merges[i]]=i
        if special_tokens is not None:
            for tok in special_tokens:
                self.special_token_ids[tok]=self.inverse_vocab[tok]

    #build from built vocabs files
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[bytes]|None=None):
        tk=cls()
        vocab:dict[int,bytes]={}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            for token_str, token_id in vocab_json.items():
                token_bytes = bytes([int(b) for b in token_str.split(",")])
                vocab[int(token_id)] = token_bytes
        merges:list[tuple[bytes,bytes]]=[]
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                token_str1, token_str2 = line.strip().split()
                token_bytes1 = bytes([int(b) for b in token_str1.split(",")])
                token_bytes2 = bytes([int(b) for b in token_str2.split(",")])
                merges.append((token_bytes1, token_bytes2))
        tk.build_from_vocab_merges(vocab,merges,special_tokens)
        return tk

    #encode single pretoken
    def encodesingle(self,word:bytes)->list[int]:
        #to wordlist
        wordlist=[bytes([b]) for b in word]
        while len(wordlist)>=2:
                #find the bestpair
                minrank=float("inf")
                bestpair=None
                for i in range(len(wordlist)-1):
                    pair=(wordlist[i],wordlist[i+1])
                    rank=self.bpe_ranks.get(pair)
                    if rank is not None and rank<minrank:
                        minrank=rank
                        bestpair=pair
                if bestpair is None:
                    break
                #create newword
                newword=[]
                i=0
                while i<len(wordlist):
                    if i<len(wordlist)-1 and wordlist[i]==bestpair[0] and wordlist[i+1]==bestpair[1]:
                        newword.append(bestpair[0]+bestpair[1])
                        i+=2
                    else:
                        newword.append(wordlist[i])
                        i+=1
                wordlist=newword
        return [self.inverse_vocab[token] for token in wordlist]

    #encode the text to vector
    def encode(self,text:str)->list[int]:
        ids=[]
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens=[re.escape(tok.decode("utf-8")) for tok in special_tokens]
            split_pattern=f"({'|'.join(escaped_tokens)})"
            splittext=re.split(split_pattern,text)
        else:
            splittext=[text]
        for textpart in splittext:
            if not textpart:
                continue
            bytetext=textpart.encode("utf-8")
            #special tokens
            if bytetext in self.special_token_ids:
                ids.append(self.special_token_ids[bytetext])
            else:
                for match in re.finditer(self.pat,textpart):
                    pretoken=match.group()
                    bytewords=pretoken.encode("utf-8")
                    wordids=self.encodesingle(bytewords)
                    ids.extend(wordids)
        return ids
    
    #encode into iterator,lazy generator
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for text in iterable:
            ids=self.encode(text)
            for id in ids:
                yield id


    #decode the vector to text
    def decode(self,ids:list[int]):
        bytelist=[]
        for token_id in ids:
            if token_id in self.vocab:
                bytelist.append(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown Token ID:{token_id}")
        joinedbytes=b"".join(bytelist)   
        return joinedbytes.decode("utf-8",errors="replace")
        
