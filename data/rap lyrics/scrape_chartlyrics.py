import requests
import os
from lxml import html
import time
import random
import io

# User-agent header to fool the website into thinking it's a real person using a browser instead of a spider
headers={"User-Agent": "Mozilla/15.0 (Windows NT 6.33932234; Win64; x64) AppleWebKit/537.1352532535 (KHTML, like Gecko) Chrome/37.0.2048.2322111341334 Safari/537.982727285"}

# This is where we're gonna scrape from
SITE = "http://www.chartlyrics.com"
LYRICS = "/apiv1.asmx/SearchLyricDirect?artist={}&song={}"
ARTIST = '99zrDx9OYUaUk7QEJ94sEw' # GUID for artist
NAME = "wutangclan"

PATH = os.path.join(SITE,"{}.aspx".format(ARTIST))

# Execute request and get dat sweet html response
r = requests.get(PATH,headers=headers)

# Parse the response into an html tree
tree = html.fromstring(r.content)

# Get all anchor tags from the response
anchors = tree.xpath('//a/@href') # Finds all a tags with target="_blank" and returns their hrefs

# Get all valid lyric urls
lyric_urls = [a for a in anchors if ARTIST in a] # Valid urls have the artist code in them

# Get the total number of songs for later
num_lyrics = len(lyric_urls)

# If we get rate-limited and kicked off for a bit we wait
# then start again at this index
restart_index = 0 # Default is 0

# Iterate over all lyric urls
for i in range(restart_index,len(lyric_urls)):

    # Get the current lyric from the list we're iterating over
    lyric_path = lyric_urls[i]

    # Get the title for later
    song_title = lyric_path.replace('/{}/'.format(ARTIST),'').replace('.aspx','').replace("+"," ")

    # Append the new lyric url to the website url and get the response
    lyric_url = SITE + lyric_path
    lyric_page = requests.get(lyric_url,headers=headers)

    # Once again, convert the response into a html tree
    lyric_tree = html.fromstring(lyric_page.content)

    # get all the text from all divs in the tree
    divs = lyric_tree.xpath('//p/text()')

    # There are a ton of divs whose only content is /r, /n, ' ' or some combination of those.
    # We remove those here
    lines = [d for d in divs if len(d.replace(' ','').replace('\r','').replace('\n','').replace('\t','')) > 0]

    # Convert lines into text
    text = u''.join(lines)

    # Skip failures
    if "bad request" not in text.lower() and len(text) > 100:
        # Save the lyrics for later ingestion
        with io.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'lyrics',NAME,'{}.txt'.format(song_title)),'w+',encoding='utf-8') as f:
            f.write(text)

        # Sanity/Progress check
        print("Downloaded song {}, completed {} of {}".format(song_title,i,num_lyrics-1))

        # High likelihood of getting rate limited so we sleep for a few seconds between requests.
        # Also prevents over-working their servers.
        # Always be considerate when scraping :)
        time.sleep(random.randint(3,8))

    
