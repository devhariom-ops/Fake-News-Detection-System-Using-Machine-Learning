import os
import pandas as pd
import random
from datetime import datetime, timedelta

# Create dataset directory if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# Seed for reproducibility
random.seed(42)

# Sample elements to build realistic real news
real_subjects = ["politicsNews", "worldnews"]
real_titles = [
    "U.S. Senate passes sweeping tax reform bill",
    "Brexit negotiations reach critical stage in Brussels",
    "North Korea launches new missile test, drawing international condemnation",
    "French President proposes new environmental regulations",
    "German Chancellor seeks coalition agreement to form government",
    "U.S. Supreme Court hears major voting rights case",
    "United Nations warns of humanitarian crisis in East Africa",
    "China announces new trade policies to boost domestic consumption",
    "British Prime Minister defends economic plan in Parliament",
    "EU regulators fine major technology firm for antitrust violations",
    "Congress approves emergency funding for disaster relief",
    "Japan steps up defense spending citing regional security concerns",
    "Iran nuclear deal faces uncertainty after U.S. policy shift",
    "South African President vows to crack down on corruption",
    "NATO allies pledge to increase defense spending by 2%",
    "House of Representatives passes bipartisan infrastructure package",
    "Canada pledges support for new global climate initiative",
    "Australia announces plans to increase clean energy capacity",
    "India implements nationwide digital tax for foreign corporations",
    "Brazil details plan to curb deforestation in the Amazon basin"
]

real_bodies = [
    "WASHINGTON (Reuters) - The United States Senate on Friday approved a major overhaul of the tax code, bringing the Republican party closer to its goal of cutting taxes for corporations and individuals. The legislation passed with a narrow majority and will now move to a conference committee to reconcile differences with the House version.",
    "BRUSSELS (Reuters) - Negotiations between the United Kingdom and the European Union have entered a critical phase as both sides attempt to draft a final agreement on trade and border arrangements post-Brexit. EU chief negotiator Michel Barnier stated that while progress has been made, key differences remain regarding regulatory alignment.",
    "SEOUL/TOKYO (Reuters) - North Korea conducted another ballistic missile test early Wednesday, drawing immediate condemnation from neighboring countries and the international community. The missile flew for approximately 40 minutes before landing in the Sea of Japan, according to Japanese defense officials.",
    "PARIS (Reuters) - French President Emmanuel Macron announced a series of new environmental policies aimed at reducing carbon emissions and promoting green energy technologies. Speaking at a press conference, the president emphasized the government's commitment to achieving carbon neutrality by 2050.",
    "BERLIN (Reuters) - Chancellor Angela Merkel urged political parties to find common ground as formal coalition talks resumed in Berlin. The country has been without a stable government since the last election, and officials warn that failure to reach an agreement could lead to fresh elections.",
    "WASHINGTON (Reuters) - The U.S. Supreme Court began hearing oral arguments today in a landmark case regarding state legislative districts. Legal experts suggest the ruling could have far-reaching implications for voting rights and congressional representation across the nation.",
    "GENEVA (Reuters) - The United Nations issued an urgent appeal to international donors on Thursday, warning that millions of people in East Africa are facing severe food shortages due to consecutive years of drought. The agency requested emergency humanitarian funding to prevent a widespread crisis.",
    "BEIJING (Reuters) - China’s Ministry of Commerce announced several trade policy updates designed to lower import tariffs on consumer goods and stimulate domestic spending. Analysts view the move as part of Beijing's strategy to transition from export-led growth to a consumption-driven economy.",
    "LONDON (Reuters) - Prime Minister Keir Starmer defended his administration’s economic strategy during a heated session of Prime Minister's Questions in the House of Commons today. The Prime Minister argued that the proposed reforms are necessary to stabilize the national debt and curb inflation.",
    "BRUSSELS (Reuters) - The European Commission has fined a major tech conglomerate €1.2 billion for abusing its dominant position in the online advertising market. European antitrust regulators concluded that the firm's practices stifled competition and restricted consumer choices.",
    "WASHINGTON (Reuters) - The U.S. Congress has approved a $15 billion emergency aid package to assist communities recovering from recent hurricanes. The bill received broad bipartisan support in both chambers and was sent to the President to be signed into law.",
    "TOKYO (Reuters) - Japan announced a significant increase in its defense budget for the upcoming fiscal year, pointing to growing security challenges in the Indo-Pacific region. Government spokespersons stated the funds would go toward upgrading naval defenses and early warning radar systems.",
    "VIENNA (Reuters) - The future of the joint nuclear agreement remains uncertain as international inspectors confirmed that Iran continues to enrich uranium. Representatives from European co-signatories met in Vienna to discuss diplomatic options to preserve the accord.",
    "JOHANNESBURG (Reuters) - South Africa's president vowed to strengthen judicial institutions and investigate high-ranking officials accused of embezzling public funds. In a national address, he declared that restoring public trust and stamping out corruption is his top priority.",
    "BRUSSELS (Reuters) - NATO member states reaffirmed their commitment to defense spending goals during a summit of defense ministers. The alliance's Secretary General noted that a record number of allies are on track to meet the target of spending 2% of GDP on defense.",
    "WASHINGTON (Reuters) - The House of Representatives passed a $1.2 trillion bipartisan infrastructure bill, funding roads, bridges, public transit, and clean water systems. The legislation represents a major policy victory for the administration after months of negotiations.",
    "OTTAWA (Reuters) - Prime Minister Justin Trudeau announced that Canada will join a new international alliance to phase out coal-fired power plants. Trudeau stated that global cooperation is vital to meet emissions targets established under the Paris Agreement.",
    "CANBERRA (Reuters) - The Australian government unveiled a new strategy to transition its electricity grid to renewable energy. The plan includes funding for major solar and wind projects, as well as investments in grid battery storage facilities.",
    "NEW DELHI (Reuters) - India’s finance ministry defended its decision to levy an equalization tax on foreign digital service providers. The ministry stated the tax aims to ensure a level playing field for domestic tech companies and collect fair revenues from digital commerce.",
    "BRASILIA (Reuters) - The Brazilian government launched a new task force comprising environmental inspectors and federal police to combat illegal logging and mining operations in the Amazon rainforest. The administration faces pressure to protect the ecosystem."
]

# Sample elements to build realistic fake news
fake_subjects = ["Government News", "Middle-east", "US_News", "left-news"]
fake_titles = [
    "ALERT: Secret government documents reveal shocking truth about voter fraud!",
    "OMG! Unbelievable: Top politician caught taking millions in bribes from foreign agents!",
    "Breaking: NASA admits alien spacecraft is heading directly toward Earth!",
    "The mainstream media won't show you this: Massive protests block city streets!",
    "SHOCKING: New medical report proves common food is highly toxic and being covered up!",
    "Caught on camera: Secret meeting exposes shadow government planning next lock-down!",
    "BREAKING NEWS: Global elites meet in secret Swiss bunker to decide currency reset!",
    "Wow: Celebrities are fleeing the country after new tax laws are proposed!",
    "Proof: Famous historical monument was actually built by extraterrestrial intelligence!",
    "Watch: Secret speech by former president warns of impending collapse of the dollar!",
    "ALERT: Doctors speak out against hidden ingredients in public water systems!",
    "Unbelievable: Local mayor bans national flags to avoid offending foreign visitors!",
    "Breaking: Famous billionaire revealed to be funding secret mind-control chips!",
    "SHOCKING: Police department ordered to ignore all emergency calls for 24 hours!",
    "Leaked video: Intelligence agency official confesses to major historical cover-up!",
    "OMG: Massive gold reserve discovered under private farm, government immediately seizes it!",
    "It's official: Scientific study reveals cell phone signals are mutating plants!",
    "Breaking: Major bank goes bankrupt overnight, government freezes all citizens' assets!",
    "Voter alert: Thousands of pre-marked ballots discovered hidden in warehouse basement!",
    "Shocking discovery: Secret underground city found beneath national park, media silent!"
]

fake_bodies = [
    "Unbelievable news reports are spreading like wildfire across social media platforms! Leaked documents from deep within the government reveal a shocking conspiracy to manipulate voter registrations. According to anonymous insiders, thousands of records were altered. The mainstream media is completely silent on this, proving they are in on the cover-up. Share this article before it gets banned!",
    "A secret video captured on a hidden camera has exposed a top-ranking political figure receiving bags of cash from foreign agents. The meeting took place in a dark restaurant alley last Tuesday. Sources confirm that millions of dollars were exchanged in exchange for secret legislative favors. The authorities have refused to comment, and the establishment is doing everything to keep this hidden.",
    "In a shocking turn of events, a leaked report from space agency scientists has confirmed that an unidentified object of immense size is currently on a direct collision course with Earth. The agency has supposedly kept this information under wraps to prevent global panic, but whistleblowers have decided to expose the truth. Experts suggest preparing for impact within weeks.",
    "You won't see this on the nightly news! A massive crowd of over a million patriotic citizens has shut down all major highways entering the capital. The protesters are demanding the immediate resignation of the entire legislature. Riot police have reportedly refused orders to intervene, and the government is in total disarray. Watch the live stream link here!",
    "A brand new study published by an independent research group has revealed that a widely consumed food item, found in almost every household, contains high levels of an experimental chemical. The chemical has been linked to severe health issues. The report claims that major food corporations paid off inspectors to keep this study from being published.",
    "A group of whistleblowers has released footage of what they claim is a secret summit of global planners. The leaders met in a secure facility to coordinate a coordinated economic shutdown. The plan is to phase out physical currency and force everyone into a digital surveillance grid. Mainstream journalists were banned from the event.",
    "Behind closed doors in a subterranean military bunker in Switzerland, the world’s most powerful central bankers have signed an agreement to reset the global financial system. According to inside reports, they plan to devalue all major currencies by 80% overnight. Citizens are advised to convert their cash into physical assets immediately.",
    "A growing list of top Hollywood actors and musicians have reportedly purchased private islands and are leaving the country. Sources close to the stars claim they are escaping the new financial auditing measures scheduled to take effect next month. Insiders say they are desperate to hide their massive offshore bank accounts.",
    "Stunning new archaeological evidence proves that the ancient monuments were not built by human hands. High-resolution scans show microscopic precision cuts that could only be achieved with high-energy laser tools. Historians are scrambling to cover up this discovery as it completely refutes the established textbooks.",
    "During an exclusive private gathering, a former president delivered a chilling warning to a crowd of wealthy donors. The speaker stated that the national currency is on the verge of complete hyperinflation and will become worthless paper by the end of the year. The audio was secretly recorded by a staff member and leaked online.",
    "A coalition of independent doctors has held a press conference warning that public utilities have been adding chemical compounds to public drinking water to regulate population behavior. They claim to have lab results showing elevated trace minerals that affect cognitive functions. Officials dismissed the claims as baseless theories.",
    "Outrageous decision! The city council of a midwestern town has voted to ban all national flags on public property, claiming they make visitors feel unwelcome. Local citizens are planning a massive recall election against the mayor, who allegedly proposed the ban during a closed-session meeting. The local media has ignored the protests.",
    "A leaked email chain has exposed a tech billionaire's plan to distribute experimental tracking technology in common medical items. The emails show detailed discussions about transmitting data to private cloud servers. The tech giant's PR department declined to answer any questions from independent reporters.",
    "A secret memo sent to all officers of a major metropolitan police force instructed them to cease all emergency responses for a full 24-hour period during a local political event. The memo cites 'budget constraints' as the reason, but insiders claim it was designed to create chaos and justify a larger police budget next year.",
    "An intelligence official has admitted on camera that several high-profile historical incidents were entirely simulated to influence public opinion. The video, uploaded to an anonymous hosting platform, shows the official explaining the logistics and scriptwriting process. The video is being rapidly deleted from social media platforms.",
    "A farmer in a rural county struck gold while drilling a water well, uncovering what geologists call the largest gold vein in the state. However, before the family could claim the find, federal officers seized the property under a forgotten 19th-century land statute. The family is currently suing for compensation.",
    "A controversial research paper claims that radio frequencies emitted by cell towers are causing genetic anomalies in local plant life. The author, who was recently dismissed from a state university, claims that trees near high-power antennas show altered growth rates. Mainstream botanists have rejected the paper's findings.",
    "Chaos erupted as one of the country's oldest investment banks shut down its online portal and locked its doors. A leaked government memo indicates that all personal deposits have been temporarily frozen to prevent a run on the banking sector. Long lines have formed outside bank branches, with police standing guard.",
    "Voter rights advocates are demanding a full audit after a delivery driver discovered boxes containing thousands of pre-filled ballots in the basement of a rented warehouse. The ballots were all marked for the same candidate. Authorities claim the boxes were discarded training materials, but critics are skeptical.",
    "A team of deep-cave explorers has mapped a massive subterranean facility under a national park, complete with power grids and housing blocks. The explorers claim the site shows signs of recent occupancy. The National Park Service has closed off access to the area, citing safety concerns due to 'geological instability.'"
]

# Generate synthetic articles
def generate_articles(titles, bodies, subjects, count):
    data = []
    start_date = datetime(2016, 1, 1)
    
    for i in range(count):
        # Pick title and body
        title_idx = i % len(titles)
        body_idx = i % len(bodies)
        
        title = titles[title_idx]
        body = bodies[body_idx]
        
        # Add random variations to avoid exact duplicates and make it more natural
        # Insert a random word or modify spacing
        words = body.split()
        if len(words) > 10:
            insert_idx = random.randint(1, len(words) - 5)
            words.insert(insert_idx, random.choice(["absolutely", "shockingly", "reportedly", "allegedly", "officially"]))
        body = " ".join(words)
        
        subject = random.choice(subjects)
        
        # Generate random date
        random_days = random.randint(0, 700)
        article_date = start_date + timedelta(days=random_days)
        date_str = article_date.strftime("%B %d, %Y")
        
        data.append({
            "title": title,
            "text": body,
            "subject": subject,
            "date": date_str
        })
        
    return pd.DataFrame(data)

# Generate 200 real and 200 fake articles
print("Generating synthetic real news...")
real_df = generate_articles(real_titles, real_bodies, real_subjects, 250)
print(f"Generated {len(real_df)} real news articles.")

print("Generating synthetic fake news...")
fake_df = generate_articles(fake_titles, fake_bodies, fake_subjects, 250)
print(f"Generated {len(fake_df)} fake news articles.")

# Save to CSV files
real_path = os.path.join("dataset", "True.csv")
fake_path = os.path.join("dataset", "Fake.csv")

real_df.to_csv(real_path, index=False)
fake_df.to_csv(fake_path, index=False)

print(f"Dataset successfully created in:")
print(f" - {real_path}")
print(f" - {fake_path}")
print("Mock dataset generation completed successfully!")
