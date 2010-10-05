$(document).ready(function() {

		$('.panel_button').hide();

       $('#friends_news_header').hide();
       $('#everyone_news_header').hide();
       $('#twitter_news_header').hide();

       $('#friends_news').hide();
       $('#everyone_news').hide();
       $('#twitter_news').hide();

	$("div.panel_button").click(function(){
		$("div#panel").animate({
			height: "600px"
		})
		.animate({
			height: "600px"
		}, "fast");
		$("div.panel_button").toggle();
		
		$('#friends_news_header').toggle();
                $('#everyone_news_header').toggle();
                $('#twitter_news_header').toggle();
		
                $('#friends_news').toggle();
                $('#everyone_news').toggle();
                $('#twitter_news').toggle();
	});	
	
   $("div#hide_button").click(function(){
		$("div#panel").animate({
			height: "0px"
		}, "fast");	
	
   });	
	
});