import torch
#from torchtext.legacy.data import Field
import torchtext as tx
from torchtext.vocab import GloVe
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
import re
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
embedding_size = 100
Train_size=30000



def review_clean(text):
    text = re.sub(r'[^A-Za-z]+', ' ', text)  # remove non alphabetic character
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove singale char
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokinize(s):
    s = review_clean(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]


def load_data_set(custom_reviews=None):
    data=pd.read_csv("IMDB Dataset.csv")
    train_data=data[:Train_size]
    train_iter=ReviewDataset(train_data["review"],train_data["sentiment"])
    test_data=data[Train_size:]
    if custom_reviews is not None:
        test_data = pd.DataFrame({"review": custom_reviews[0], "sentiment": custom_reviews[1]})
    test_data=test_data.reset_index(drop=True)
    test_iter=ReviewDataset(test_data["review"],test_data["sentiment"])
    return train_iter, test_iter


embadding = GloVe(name='6B', dim=embedding_size)
tokenizer = get_tokenizer(tokenizer=tokinize)


def preprocess_review(s):
    cleaned = tokinize(s)
    embadded = embadding.get_vecs_by_tokens(cleaned)
    if embadded.shape[0] != 100 or embadded.shape[1] != 100:
        embadded = torch.nn.functional.pad(embadded, (0, 0, 0, MAX_LENGTH - embadded.shape[0]))
    return torch.unsqueeze(embadded, 0)


def preprocess_label(label):
    return [0.0, 1.0] if label == "negative" else [1.0, 0.0]


def collact_batch(batch):
    label_list = []
    review_list = []
    embadding_list=[]
    for  review,label in batch:
        label_list.append(preprocess_label(label))### label
        review_list.append(tokinize(review))### the  actuall review
        processed_review = preprocess_review(review).detach()
        embadding_list.append(processed_review) ### the embedding vectors
    label_list = torch.tensor(label_list, dtype=torch.float32).reshape((-1, 2))
    embadding_tensor= torch.cat(embadding_list)
    return label_list.to(device), embadding_tensor.to(device) ,review_list


##########################
# ADD YOUR OWN TEST TEXT #
##########################

# Test texts for experiment 4, for re-predicting same texts used as TP/FP/TN/FN in experiment 2
exp4_texts = []
exp4_texts.append("There are a few aspects to Parks movies, and in particular Wallace & Gromit, that I would say make them so great. The first is subtlety and observation, the flagship of which is the character of Gromit. He doesn't speak, he doesn't make any noise, all he has are his eyes, brow, and body posture, and with these he commands the film. Park manages to give us everything we need from this silent character through his expression. The comedy and the emotion is conveyed through the subtlest of movements and it works superbly well.<br /><br />Watching the movie you have to be aware of the entire screen. Normally you'll be guided to things in the movies, the screen won't be cluttered too much, there won't be many things to take your eyes away from the main clue or action. Park seems to need to look the other way with his movies. He throws extra content at his audience, there's action in the background, to the side of the screen, even off screen, and there's just about always something in the foreground to catch your eye. His movies are about multiple viewing and discovery, they're layered with jokes and ancillary action.<br /><br />Throughout this film there are layers of things happening on screen, jokes in the foreground maybe on a jar label and background shadows that give away action. You can imagine that for Park the movies has always been an event, and the movies he loves are ones which he wants to watch again and again. This is what shows in his movies, and in through his most beloved characters.<br /><br />Then there are the bizarre and wacky inventions which Wallace make, something which is reflected in the storyline and the twists and turns of the plot, everything is bizarre and off the wall, yet it seems so perfectly normal in this world. You can imagine that inside Park is the mind of Wallace.<br /><br />There's also one more thing that make these movies so unique, and that's the modelling and precise hand animation. I must admit I was concerned when I knew Dreamworks was involved in the making of this movie, and I thought that they would bring their computer animation experience to the forefront. What I was scared of was Wallace & Gromit becoming CGI entities, or at the smallest, CGI being used to clean up the feel that the modelling brought to the movie.<br /><br />Not so. You can still see thumbprints and toolmarks on the characters, and far from distracting from the movie, this just adds so much real feeling to it and a feeling of physical depth to the characters and the scene on screen.<br /><br />So what of the movie? Well I must say that the plot twist was something I had thought about well before the film was in the cinema and it came as no surprise, but that did not affect my enjoyment one little bit. Actually watching the twist unfold and the comic timing of the discovery and reactions was everything, and it had me just as sucked in as if it was a thriller, yet all the time I was laughing.<br /><br />Watching the movie was fascinating in various ways. To see the animation completed, how wild the inventions are, how Wallace is going to get into trouble and Gromit get him out, where all the cross references are in the movie, and where all the jokes are! I must admit afterwards talking with my friends I couldn't believe how much I had missed.<br /><br />There's something different in this movie than with the others, there's a new level of adult humour in here, and I don't mean rude jokes (although there are a couple that are just so British you can't help laughing), I mean jokes that simply fly over kids heads but slap adults in the face. The kind you are used to seeing come out of somewhere like Pixar. This just adds even more appeal to the movie.<br /><br />Okay though, let me try and be a bit negative here. I didn't notice the voices in this movie, you know how you usually listen to the actors and see if you can recognise them? Well I was just too wrapped up in the movie to care or to notice who they were...okay, thats not negative. Let me try again. The main plot wasnt as strong and gripping as Id expected, and I found myself being caught up in the side stories and the characters themselves...again...thats not a bad thing, the film was just so much rich entertainment.<br /><br />I honestly cant think of a bad thing to say about this movie, probably the worst thing I could say is that the title sequence at the end is quite repetitive...until the final title! Really, thats the worst I can say.<br /><br />The story is a lot of fun, well set-up, well written, well executed. Theres lots of fantastic characters in here, not just Wallace & Gromit. Theres so much happening on screen, so many references and jokes (check out the dresses of Lady Tottingham), cheese jokes everywhere, jokes for all the family. The characters are superbly absorbing and youll find that youve taken to them before you realise. Theres just so much in this movie for everyone.<br /><br />Theres so much I could say and write about, but I know it will quickly turn into a backslapping exercise for Park and Aardman, it would also just turn into a series of this bit was really funny and theres a bit when..., and what I would rather do is tell you that this is a superb movie, to go see it, and to experience the whole thing for yourselves. I will say though that the bunnies are excellent!")
exp4_texts.append("...and not in a good way. BASEketball is a waste of film in all most every single way. It is offensive to all the senses. This doesn't necessarily bother me, I've seen plenty of bad movies, really bad movies before and will see them again. BASEketball though is a caliber film where you regret wasting ninety minutes of life sitting through it. The reason BASEketball offends me is that it stars Trey Parker and Matt Stone in a film they didn't write. Any respect I had for David Zucker has long since depleted. His recent spoof films are lazy messes that look and feel as if they were made by pre-pubescent boys snickering at penis jokes. Airplane was a revolutionary and very funny comedy, watching BASEketball you will be amazed to discover that they were made by the same person.<br /><br />I have so much respect for Trey Parker and Matt Stone. These men are the funniest and smartest comedians in mainstream entertainment today. Their pictures and South Park episodes are as relevant as they are funny. Every joke even the fart jokes have intelligence behind them. It's easy to forget that there is a mature way to approach immaturity. I imagine BASEketball was a major growing experience for them because they hate the film for all the right reasons. It is a stupid mess with no sense of dignity or class. Parker and Stone have essentially whored themselves out. The film plays like a 90 minute episode of Family Guy.<br /><br />Parker and Stone have never been great actors. They've been serviceable in their films. I can't really find a way to describe their performance in BASEketball, other than the fact that it feels like they are spoofing a spoof film spoofing a spoof film. Every line is delivered in such a silly winking way. It's like they are trying to make fun of the worst of these type of pictures and yet they become them in the same way. I am reminded of the South Park episode How to Eat with your Butt where Cartman sits in a movie theater watching a gross out comedy with no plot or plausibility except to gross out, Parker and Stone use the same voices they did in that scene for this entire picture. Really it's sad.<br /><br />And yet that is not my problem with BASEketball. My biggest gripe with the picture is that I sit there knowing that Parker and Stone are knowingly following this piece of crap script. I know that if they took the damn thing and rewrote it that this could have been salvaged to the point of being watchable. There isn't any indication that Zucker let them improv scenes either. Parker and Stone are merely tools to a bad director. BASEketball has some funny concepts and I think Parker especially if he were allowed to take Zuckers script could have elaborated on them more. Instead we get potty humor. Don't rent BASEketball you can get the same laughs watching a group of grade schoolers joking around")
exp4_texts.append("This film features two of my favorite guilty pleasures. Sure, the effects are laughable, the story confused, but just watching Hasselhoff in his Knight Rider days is always fun. I especially like the old hotel they used to shoot this in, it added to what little suspense was mustered. Give it a 3.")
exp4_texts.append("There's no other word for it...Fox dumped this out, with NO marketing of any kind. Nobody in the country, other than those who have been looking forward to this film, know anything about it. All the red flags have flown. It has to be a mess, it can't be anywhere near as good as Office Space, right? Wrong. Though Office Space it ain't, this film definitely has satirical bite and wit. It's a misfire on certain levels, but who's to blame is left to mystery.<br /><br />Based on what is currently showing in theatres, I can say IDIOCRACY is a good movie. It's funny, sometimes laugh-out-loud funny. It's effective, sometimes ingenious. What it isn't as far as I can tell, is finished. We will see something come of this film again, whether it's an extended cut or reshoots. Alone it can be hilarious. It's ballsy at times.<br /><br />Leaving the theatre, looking around at the mall, I was surrounded by advertisements and billboards, commercialism and stupidity. It's not quite as damning a dystopia as 1984, but this movie paints an ugly future for our culture. And there doesn't seem to be much anybody can do about it. Anyway, go see this if you can and try to find out what happened that it was so specifically buried.")
exp4_labels = ["positive", "negative", "negative", "positive"]

my_test_texts = []
# Aiming for True-Negative on GRU, and False-Positive on RNN
my_test_texts.append("Director pursued for greatest, most amazing movie ever. But it's bad.")
# Aiming for False-Positive on both Pure MLP & MLP-Atten
my_test_texts.append("The only great thing about this movie is that it ended quickly")
# Aiming for False-Negative on both Pure MLP & MLP-Atten
my_test_texts.append("The movie follows a group of the worst, most horrible humans, so creative!") 
# Aiming for False-Positive on Pure MLP, and True-Negative on MLP-Atten
my_test_texts.append("The greatest disappointment of recent history")
my_test_labels = ["negative", "negative", "positive", "negative"]

##########################
##########################


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, review_list, labels):
        'Initialization'
        self.labels = labels
        self.reviews = review_list

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        X = self.reviews[index]
        y = self.labels[index]
        return X, y



def get_data_set(batch_size, custom_reviews=None):
        train_data, test_data = load_data_set(custom_reviews)
        train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                      shuffle=True, collate_fn=collact_batch)
        test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                     shuffle=True, collate_fn=collact_batch)
        return train_dataloader, test_dataloader, MAX_LENGTH, embedding_size


