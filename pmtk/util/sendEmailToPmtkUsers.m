
emailList = parseDownloads('C:\kmurphy\PMTKlocal\PMTKdownloads.txt');
%emailList = {'murphyk@cs.ubc.ca', 'mattdunham@yahoo.com'};
subject = 'PMTK email list';
%message = 'Please visit http://www.cs.ubc.ca/pmtk to download the latest version';
message = 'This is a test of the pmtk broadcasting system';
sendmail(emailList,subject,message);