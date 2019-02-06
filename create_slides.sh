if [ "$#" -ne 0 ]; then
  echo "$#"
  jupyter nbconvert notebook.ipynb --to slides --reveal-prefix $1
else
  echo 'provide location of reveal.js'
fi
