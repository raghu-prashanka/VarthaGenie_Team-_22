# -- coding: utf-8 --
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the saved model and tokenizer
model_name = './telugu_summary_model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to summarize text
def summarize(text):
    print("Input Text:", text)  # Debug: Print input text

    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding="max_length")
    print("Tokenized Inputs:", inputs)  # Debug: Print tokenized inputs

    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    print("Summary IDs:", summary_ids)  # Debug: Print generated summary IDs

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Decoded Summary:", summary)  # Debug: Print decoded summary
    return summary

# Example usage
text = "తెలంగాణ ముఖ్యమంత్రి కల్వకుంట్ల చంద్రశేఖర్‌రావు రానున్న అసెంబ్లీ ఎన్నికల్లో రెండు నియోజకవర్గాల నుంచి పోటీ చేయనుండడం రాష్ట్రవ్యాప్తంగా ఆసక్తి రేపుతోంది. ఆయన తన సిటింగ్ స్థానం గజ్వేల్ నుంచే కాకుండా సమీపంలోని కామారెడ్డి నియోజకవర్గం నుంచి కూడా పోటీ చేయనున్నారు. బీఆర్ఎస్ పార్టీ తొలి విడతలో 115 అసెంబ్లీ స్థానాలకు ప్రకటించిన అభ్యర్థుల జాబితా ప్రకారం కేసీఆర్ రెండు నియోజకవర్గాల నుంచి పోటీ చేయడం ఖరారైనట్లే. అయితే, ఏ అభ్యర్థి అయినా రెండు చోట్ల నుంచి పోటీ చేసి రెండు చోట్లా గెలిస్తే ఓ స్థానాన్ని వదులుకోవాల్సి ఉంటుంది. అప్పుడు ఆ నియోజకవర్గానికి మళ్లీ ఎన్నికలు నిర్వహించాల్సిందే. దీంతో ఇలా ఒకటి కంటే ఎక్కువ స్థానాల నుంచి పోటీ చేయడం వల్ల ఖజానాకు నష్టం ఏర్పడుతుందని, ప్రజాధనం వృథా అవుతుందని, ఎన్నికల నిర్వహణకు మళ్లీ విలువైన మానవ వనరులు కూడా వృథా అవుతాయన్న వాదన చాలా కాలంగా ఉంది. అంతేకాదు.. ఒక ఓటరు ఒక అభ్యర్థికి మాత్రమే ఓటు వేయాలన్న నిబంధన ఉన్నప్పుడు అభ్యర్థులు రెండు నియోజకవర్గాల నుంచి పోటీ చేయడం సరైనదేనా అనే చర్చ కూడా ఉంది. అయితే, ఇలా రెండు నియోజకవర్గాల నుంచి పోటీ చేస్తున్నది కేసీఆర్ ఒక్కరే కాదు. ఇంతకుముందు నరేంద్ర మోదీ, రాహుల్ గాంధీ, ఇందిరా గాంధీ, ఎన్టీఆర్, పీవీ నరసింహారావు వంటి ఎందరో నాయకులు ఒకటి కంటే ఎక్కువ నియోజకవర్గాల నుంచి పోటీ చేశారు. ఇంతకూ భారత దేశంలో శాసనసభ, లోక్‌సభ ఎన్నికలలో ఒక అభ్యర్థి గరిష్ఠంగా ఎన్ని స్థానాలలో పోటీ చేయొచ్చు? ప్రజా ప్రాతినిధ్య చట్టం ఏం చెప్తోంది? గతంలో ఎలక్షన్ కమిషన్ ఈ విషయంలో ఎలాంటి సిఫారసులు చేసింది? లా కమిషన్ ఎలాంటి సూచనలు చేసింది? సుప్రీంకోర్టు ఏం చెప్పింది?"

summary = summarize(text)
print("Summary:", summary)