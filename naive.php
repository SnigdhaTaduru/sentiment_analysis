<!DOCTYPE html>
<html>
    <head>
        <title>NaiveBayes</title>
        <link href="https://fonts.googleapis.com/css?family=Acme" rel="stylesheet">
        <style>
            body {
                background-image: url(background.jpg);
                color: floralwhite;
            }
            header .heading
            {
                font-family: 'Acme', sans-serif;
                font-size: 300%;
            }    
        </style>
    </head>
    <body>
        <header>
            <h1 class='heading'><center>Naive Bayes Results</center></h1> 
        </header>
        <?php
        $pyscript="naivebayes.py";
        $cmd="$pyscript";
        $naiveaccuracy=exec("$cmd");
        echo $naiveaccuracy
        ?>
    </body>
</html>