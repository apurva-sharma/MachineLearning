<?php
  $file = fopen("wrds_new.txt", "r") or exit("Unable to open file!");
  //Output a line of the file until the end is reached
  $i = 0;
  $results = '';
  while(!feof($file))
  {
	$l = fgets($file);
	$t = explode('|', $l);
	$results = $results . $t[0] . ',' . $t[1] . ',';
  }
fclose($file);
echo $results;
?>