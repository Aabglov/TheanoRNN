import requests
import os
from lxml import html
from lxml import etree
import time
import random
import io

# User-agent header to fool the website into thinking it's a real person using a browser instead of a spider
headers={"User-Agent": "Mozilla/15.0 (Windows NT 6.3234; Win64; x64) AppleWebKit/537.135235 (KHTML, like Gecko) Chrome/37.0.2048.23221334 Safari/537.982285"}

# This is where we're gonna scrape from
SITE = "http://www.chartlyrics.com"
LYRICS = "/apiv1.asmx/SearchLyricDirect?artist={}&song={}"
API_URL = "http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect?artist={}&song={}"
ARTIST = 'eM8r-GlbIkal73OAB2jZrA' # GUID for artist
NAME = "jay z"

PATH = os.path.join(SITE,"{}.aspx".format(ARTIST))

# Execute request and get dat sweet html response
r = requests.get(PATH,headers=headers)

# Parse the response into an html tree
tree = html.fromstring(r.content)

# Get all anchor tags from the response
anchors = tree.xpath('//a/@href') # Finds all a tags with target="_blank" and returns their hrefs

# Get all valid lyric urls
lyric_urls = [a for a in anchors if ARTIST in a] # Valid urls have the artist code in them
lyric_urls = [l for l in lyric_urls if l[:4] != "http"] # Valid urls aren't external links

# Get the total number of songs for later
num_lyrics = len(lyric_urls)

# If we get rate-limited and kicked off for a bit we wait
# then start again at this index
restart_index = 38 # Default is 0

# Iterate over all lyric urls
for i in range(restart_index,len(lyric_urls)):

    # Get the current lyric from the list we're iterating over
    lyric_path = lyric_urls[i]

    # Get the title for later
    song_title = lyric_path.replace('/{}/'.format(ARTIST),'').replace('.aspx','').replace("+"," ").replace(".","").replace(",","")

    # Append the new lyric url to the website url and get the response
    #lyric_url = SITE + lyric_path
    # UPDATING TO USE API PAGE INSTEAD OF REGULAR PAGE
    lyric_url = API_URL.format(NAME,song_title)
    lyric_page = requests.get(lyric_url,headers=headers)

    # Ensure lyrics were found
    if lyric_page.content != b'SearchLyricDirect: No valid words left in contains list, this could be caused by stop words.\r\n':

        # Once again, convert the response into a html tree
        #lyric_tree = html.fromstring(lyric_page.content)
        # USING XML IN REST API
        lyric_tree = etree.fromstring(lyric_page.content)
        for r in lyric_tree:
            if r.tag == "{http://api.chartlyrics.com/}Lyric":
                text = r.text

        # Only proceed if text was found for the lyric
        if text is not None:
            divs = text.split("\n")

            # USING XML IN REST API
            # get all the text from all divs in the tree
            #divs = lyric_tree.xpath('//p/text()')
            
            # There are a ton of divs whose only content is /r, /n, ' ' or some combination of those.
            # We remove those here
            lines = [d for d in divs if len(d.replace(' ','').replace('\r','').replace('\n','').replace('\t','')) > 0]

            # Convert lines into text
            text = u'\n'.join(lines)

            # Skip failures
            if "bad request" not in text.lower() and len(text) > 100:
                # Save the lyrics for later ingestion
                with io.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'lyrics',NAME,'{}.txt'.format(song_title)),'w+',encoding='utf-8') as f:
                    f.write(text)

                # Sanity/Progress check
                print("Downloaded song {}, completed {} of {}".format(song_title,i,num_lyrics-1))

                # Chartlyrics uses a 20 second governor.  Bummer
                time.sleep(20)
            else:
                print("Bad request, killing...")
                break

        
