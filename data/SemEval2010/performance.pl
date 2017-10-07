#!/usr/bin/perl

# for test, use 'testanswer.txt

# total test paper is 100
my $paperNumber = 100;

# read author keyphrases
my $testpapername = shift;
#print "testpapername=".$testpapername.":\n";exit;

my %CANDIDATE = ();
my $candidateCnt = 0;
open(FILE, $testpapername) or die $!;
while(my $line = <FILE>){
  chomp($line);
  my ($papername, $keystr) = split(/ : /, $line);
  $CANDIDATE{$papername} = $keystr;
  $candidateCnt++;
}

MeasurePerformance("author");
MeasurePerformance("reader");
MeasurePerformance("combined");

sub MeasurePerformance
{
  my $current_set = shift;
  my $current_answerset = "test.".$current_set.".final";

#print "filename=".$current_answerset."\n";exit;
  # store performances for all papers
  my @Correct = undef;
  my @Precision = undef;
  my @Recall = undef;
  my @Fscore = undef;
  my @Match = undef;
  my $CorrectKeyword = 0; # total number of answer keyphrases

  # initialize the buffers
  for(my $i = 0; $i < 100; $i++){
    $Correct[$i] = $Precision[$i] = $Recall[$i] = $Fscore[$i] = $Match[$i] = 0;
  }

  # read test.author/reader/combined
  open(FILE, $current_answerset) or die $!;
  my $cnt = 0;
  while(my $line = <FILE>){
    chomp($line);
    # get keyphrases of a paper in test/author/reader/combined
    my ($papername, $keystr) = split(/ : /, $line);
    my @answerset = split(/\,/, $keystr);
    my $localKeynum = scalar(@answerset);
    $CorrectKeyword += $localKeynum;

    # get keyphrases of a paper in candidiates
    my $tmpcandidate = $CANDIDATE{$papername};
    my @candidateset = split(/,/, $tmpcandidate);

    my $correct = 0;
    my $localP = 0;
    my $localR = 0;
    my $localF = 0;
    my $ind = 0;
    foreach my $one (@candidateset){ # totally 15 candidates
      foreach my $two (@answerset){
        if($two =~ /\+/){ # for alternation
          my ($first, $second) = split(/\+/, $two);
          if($first eq $one || $second eq $one){
            $correct++;
          }
        }else{
          if($one eq $two){
            $correct++;
          }
        }
      }
      $ind++;

      if($ind == 5){
        $Correct[0] += $correct;
        $localP = $correct/5;
        $localR = $correct/$localKeynum;
        if(($localP + $localR) > 0){
          $localF = 2* ($localP * $localR) / ($localP + $localR);
        }

        $Precision[0] += $localP;
        $Recall[0] += $localR;
        $Fscore[0] += $localF;
#print "5thPRF=".$correct."\t".$localP."\t".$localR."\t".$localF."\n";
      }elsif($ind == 10){
        $Correct[1] += $correct;
        $localP = $correct/10;
        $localR = $correct/$localKeynum;
        if(($localP + $localR) > 0){
          $localF = 2* ($localP * $localR) / ($localP + $localR);
        }

        $Precision[1] += $localP;
        $Recall[1] += $localR;
        $Fscore[1] += $localF;
#print "10thPRF=".$correct."\t".$localP."\t".$localR."\t".$localF."\n";
      }elsif($ind == 15){
        $Correct[2] += $correct;
        $localP = $correct/15;
        $localR = $correct/$localKeynum;
        if(($localP + $localR) > 0){
          $localF = 2* ($localP * $localR) / ($localP + $localR);
        }

        $Precision[2] += $localP;
        $Recall[2] += $localR;
        $Fscore[2] += $localF;
#print "15thPRF=".$correct."\t".$localP."\t".$localR."\t".$localF."\n";
      }
    }
    $cnt++;
#print "G5th=".$Correct[0]."\t".$Precision[0]."\t".$Recall[0]."\t".$Fscore[0]."\n";
#print "G10th=".$Correct[1]."\t".$Precision[1]."\t".$Recall[1]."\t".$Fscore[1]."\n";
#print "G15th=".$Correct[2]."\t".$Precision[2]."\t".$Recall[2]."\t".$Fscore[2]."\n";
#print "------------------------------------------------\n";

  }

#print "correct keyword = ".$CorrectKeyword."\n";
print "You answered ".$candidateCnt." over 100 test document\n";

  # print performance of test.author/reader/combined
  #................................................
  my $matchNo5 = sprintf("%d", $Correct[0]);
  my $matchNo10 = sprintf("%d", $Correct[1]);
  my $matchNo15 = sprintf("%d", $Correct[2]);

  #................................................
  # macro average
  my $Prec5 = sprintf("%.2f\%", ($Precision[0]/$paperNumber) *100);
  my $Prec10 = sprintf("%.2f\%", ($Precision[1]/$paperNumber) *100);
  my $Prec15 = sprintf("%.2f\%", ($Precision[2]/$paperNumber) *100);
  my $Recall5 = sprintf("%.2f\%", ($Recall[0]/$paperNumber)*100);
  my $Recall10 = sprintf("%.2f\%", ($Recall[1]/$paperNumber)*100);
  my $Recall15 = sprintf("%.2f\%", ($Recall[2]/$paperNumber)*100);
  my $Fscore5 = sprintf("%.2f\%", ($Fscore[0]/$paperNumber)*100);
  my $Fscore10 = sprintf("%.2f\%", ($Fscore[1]/$paperNumber)*100);
  my $Fscore15 = sprintf("%.2f\%", ($Fscore[2]/$paperNumber)*100);

  #................................................
  # micro average
  my $Prec5b = sprintf("%.2f\%", ($Correct[0]/(5*$paperNumber)) *100);
  my $Prec10b = sprintf("%.2f\%", ($Correct[1]/(10*$paperNumber)) *100);
  my $Prec15b = sprintf("%.2f\%", ($Correct[2]/(15*$paperNumber)) *100);
  my $Recall5b = sprintf("%.2f\%", ($Correct[0]/$CorrectKeyword)*100);
  my $Recall10b = sprintf("%.2f\%", ($Correct[1]/$CorrectKeyword)*100);
  my $Recall15b = sprintf("%.2f\%", ($Correct[2]/$CorrectKeyword)*100);
  my $Fscore5b = sprintf("%.2f\%", 2*($Prec5b * $Recall5b) / ($Prec5b + $Recall5b));
  my $Fscore10b = sprintf("%.2f\%", 2*($Prec10b * $Recall10b) / ($Prec10b + $Recall10b));
  my $Fscore15b = sprintf("%.2f\%", 2*($Prec15b * $Recall15b) / ($Prec15b + $Recall15b));

  print "-------------------------------------------------------------------------------------\n";
  print "[".uc($current_set)."] Match_Precision_Recall_Fscore\n";
  #print "Top 05:\t".$matchNo5."\t".$Prec5."\t".$Recall5."\t".$Fscore5."\n";
  #print "Top 10:\t".$matchNo10."\t".$Prec10."\t".$Recall10."\t".$Fscore10."\n";
  #print "Top 15:\t".$matchNo15."\t".$Prec15."\t".$Recall15."\t".$Fscore15."\n";
  print "Top 05:\t".$matchNo5."\t".$Prec5b."\t".$Recall5b."\t".$Fscore5b."\n";
  print "Top 10:\t".$matchNo10."\t".$Prec10b."\t".$Recall10b."\t".$Fscore10b."\n";
  print "Top 15:\t".$matchNo15."\t".$Prec15b."\t".$Recall15b."\t".$Fscore15b."\n";
}

