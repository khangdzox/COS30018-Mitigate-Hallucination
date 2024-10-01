import torch
from selfcheckgpt.modeling_mqag import MQAG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
mqag_model = MQAG(
    g1_model_type='race', # race (more abstractive), squad (more extractive)
    device=device
)
document = r"""
World number one Novak Djokovic says he is hoping for a "positive decision" to allow him
to play at Indian Wells and the Miami Open next month. The United States has extended
its requirement for international visitors to be vaccinated against Covid-19. Proof of vaccination
will be required to enter the country until at least 10 April, but the Serbian has previously
said he is unvaccinated. The 35-year-old has applied for special permission to enter the country.
Indian Wells and the Miami Open - two of the most prestigious tournaments on the tennis calendar
outside the Grand Slams - start on 6 and 20 March respectively. Djokovic says he will return to
the ATP tour in Dubai next week after claiming a record-extending 10th Australian Open title
and a record-equalling 22nd Grand Slam men's title last month.""".replace("\n", "")
summary = "Djokvic might be allowed to play in the US next month. Djokovic will play in Qatar next week after winning his 5th Grand slam."
score = mqag_model.score(candidate=summary, reference=document, num_questions=1, verbose=True)

print("KL-div    =", score['kl_div'])
print("Counting  =", score['counting'])
print("Hellinger =", score['hellinger'])
print("Total Var =", score['total_variation'])