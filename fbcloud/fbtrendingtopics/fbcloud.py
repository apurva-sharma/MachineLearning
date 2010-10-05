# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="akash"
__date__ ="$Sep 10, 2010 6:31:38 PM$"


#FACEBOOK_APP_ID = "114264170101"
#FACEBOOK_APP_SECRET = "3dac499fdd9f2fba339aee5b89a2e6f2"
FACEBOOK_APP_ID = "149564608409488"
FACEBOOK_APP_SECRET = "536dd97a500f49c5efa9218051728475"

import datetime
import facebook
import os
import os.path
import wsgiref.handlers
#import json

#from nltk.stem.wordnet import WordnetStemmer
from nltk.tokenize.regexp import RegexpTokenizer
from google.appengine.ext import db
from google.appengine.ext import webapp
from google.appengine.ext.webapp import util
from google.appengine.ext.webapp import template


class User(db.Model):
    id = db.StringProperty(required=True)
    created = db.DateTimeProperty(auto_now_add=True)
    updated = db.DateTimeProperty(auto_now=True)
    name = db.StringProperty(required=True)
    profile_url = db.StringProperty(required=True)
    access_token = db.StringProperty(required=True)


class BaseHandler(webapp.RequestHandler):
    """Provides access to the active Facebook user in self.current_user

    The property is lazy-loaded on first access, using the cookie saved
    by the Facebook JavaScript SDK to determine the user ID of the active
    user. See http://developers.facebook.com/docs/authentication/ for
    more information.
    """
    @property
    def current_user(self):
        """Returns the active user, or None if the user has not logged in."""
        if not hasattr(self, "_current_user"):
            self._current_user = None
            cookie = facebook.get_user_from_cookie(
                self.request.cookies, FACEBOOK_APP_ID, FACEBOOK_APP_SECRET)
            if cookie:
                # Store a local instance of the user data so we don't need
                # a round-trip to Facebook on every request
                user = User.get_by_key_name(cookie["uid"])
                if not user:
                    graph = facebook.GraphAPI(cookie["access_token"])
                    profile = graph.get_object("me")
                    user = User(key_name=str(profile["id"]),
                                id=str(profile["id"]),
                                name=profile["name"],
                                profile_url=profile["link"],
                                access_token=cookie["access_token"])
                    user.put()
                elif user.access_token != cookie["access_token"]:
                    user.access_token = cookie["access_token"]
                    user.put()
                self._current_user = user
        return self._current_user

    @property
    def graph(self):
        """Returns a Graph API client for the current user."""
        if not hasattr(self, "_graph"):
            if self.current_user:
                self._graph = facebook.GraphAPI(self.current_user.access_token)
            else:
                self._graph = facebook.GraphAPI()
        return self._graph

    def render(self, path, **kwargs):
        args = dict(current_user=self.current_user,
                    facebook_app_id=FACEBOOK_APP_ID)
        args.update(kwargs)
        path = os.path.join(os.path.dirname(__file__), "templates", path)
        self.response.out.write(template.render(path, args))


class HomeHandler(BaseHandler):
    def get_text(self, row):
        #print json.dumps(row,separators=(',',':'),indent=4)
        #print json.loads(row)
        text_list = []

        if 'message' in row:
            try:                
                text_list.append(row['message'])
            except:
                print 
        if 'description' in row:
            try:                
                text_list.append(row['description'])
            except:
                print

        return text_list

    def print_text(self, row):
        try:
            print row
        except:
            print 
                
        
    def get(self):
        if not self.current_user:
            self.render("index.html")
            return
        try:            
		#self.response.out.write("NEWS_FEED length => ")
            	news_feed = self.graph.get_connections("me", "home")
		#self.response.out.write(news_feed)
        except facebook.GraphAPIError:
	    self.response.out.write("first error")
            print "error in graph"
            self.render("index.html")
            return
        except:
	    self.response.out.write("second error")
            print "error here"
            news_feed = {"data": []}
        #print json.dumps(news_feed, separators=(',',':'))
        #print len(news_feed["data"])
	#self.response.out.write(len(news_feed))
        final_list = map(lambda x:self.get_text(x),news_feed['data'])

        #print len(final_list)
  
	
        for i in range(0,7):
            next_page = news_feed['paging']['next']

            #print next_page

            try:
                news_feed = self.graph.get_more(next_page);
            except facebook.GraphAPIError:
                print "error in graph"
                self.render("index.html")
                return
            except:
                print "error here"
                news_feed = {"data": []}

            #print len(news_feed["data"])

            final_list += map(lambda x:self.get_text(x),news_feed['data'])

            #print len(final_list)
	

        final_list=filter(lambda x:len(x)>0,final_list)
        
        final_list = map(lambda x:x[0],final_list)
	final_string=''
	for i in final_list:
            final_string+=i+','
	#self.response.out.write("FINAL LIST => ")
	#self.response.out.write(final_list)
	#self.response.out.write("FINAL STRING => ")
	#self.response.out.write(final_string)
        frequency = self.frequencyGenerator(final_string)

    def cleanedWords(self, bagofWords):
        f=open('stopList.txt')
        stopWords = f.read().strip().split(',')
        f.close()
        #print stopWords
        cleanedBag = filter(lambda x:x not in stopWords,bagofWords)
        return cleanedBag

    def frequencyGenerator(self, s):
        
        pat = '[0-9|.| |\-|\[|\]|-|!|,|\\n|\\|/|:|"|(|)|=|<|>|@|\'|#]'

        tokenizer= RegexpTokenizer(pat,gaps=True)
        allWords = tokenizer.tokenize(s)
        #stemmer = WordnetStemmer()
        allWords = map(lambda x:x.lower().strip(),allWords)
        #allWordsStemmed = map(lambda x: stemmer.lemmatize(x),allWords)        
        #del(allWords)
        allWords = self.cleanedWords(allWords)
        allWordsStemmed = allWords

        allWordsStemmed = filter(lambda x:len(x)>2,allWordsStemmed)
        #allWordsStemmed = filter(lambda x:len(x)>2,map(lambda x: stemmer.lemmatize(x),allWords))
        
        dic={}
        for i in allWordsStemmed:
            if dic.has_key(i):
                dic[i] = dic[i]+1
            else:
                dic[i]= 1

        st=''
        dic=sorted(dic.items(), key=lambda(k,v):(v,k),reverse=True)

        for k in dic:
            try:
                st+=str(k[0])+','+str(k[1])+','
            except:
                pass

	#self.response.out.write("it really does")
	#self.response.out.write("ST length => ")
	#self.response.out.write(len(st))
	#self.response.out.write("DIC length => ")
	#self.response.out.write(len(dic))
	self.response.out.write(st)
        #print st
        #return st


class PostHandler(BaseHandler):
    def post(self):
        message = self.request.get("message")
        if not self.current_user or not message:
            self.redirect("/fbcloud")
            return
        try:
            self.graph.put_wall_post(message)
        except:
            pass
        self.redirect("/fbcloud")


def main():    
#    debug = os.environ.get("SERVER_SOFTWARE", "").startswith("Development/")
    util.run_wsgi_app(webapp.WSGIApplication([
        (r"/fbcloud", HomeHandler),
        (r"/fbcloud", PostHandler),
    ], debug=False))


if __name__ == "__main__":
    main()
