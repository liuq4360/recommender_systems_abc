# A simple implementation of the RMSE calculation used for the Netflix Prize
use strict;
use warnings;
my $numValues = 0;
my $sumSquaredValues = 0;
while (<DATA>) {
    my ($rating,$prediction) = split(/\,/);
    my $delta = $rating - $prediction;
    $numValues++;
    $sumSquaredValues += $delta*$delta;
}
# Let perl do the truncation to 0.0001
printf "%d pairs RMSE: %.5f\n", $numValues, sqrt($sumSquaredValues/$numValues);

# Some example data rating & prediction data
# NOTE: This is NOT in the proper prize format
__DATA__
2,3.2
3,3.1
4,5.0
