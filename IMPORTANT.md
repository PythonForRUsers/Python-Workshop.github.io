# Important information for session development!

If you're making a session:
Put all needed files in a folder titled "session#" where # is the session number. **EXCEPT FOR FILL-IN-THE-BLANK quarto or jupyter files! Those go in the "FollowAlong" folder!!!**  
In the YAML header, make sure you have `'output-file: "session#.html"'` under the `html:` section. Again, # is the session number, or it can be anything as long as the file name starts with "session". **If you don't want to mess with the YAML header, you can name the original Quarto or Jupyter document with "session_" and the HTML file should have the same name. If you want the style to look like mine, make sure to reference `"../styles.css"` in the YAML header (since it is in the directory above the `.qmd` file).**

### How to Modify the YAML Header:

- **Change the output file**:  
  To change the name of the output file, add this under `html:` in the YAML header:
  ```yaml
  ---
  title: "Session #"
  output-file: "session#.html"  # Ensure the file starts with 'session'
  format:
    html:
      css: ../styles.css  # Reference the stylesheet from the directory above
  ---
  ```

  - Replace `#` with the session number, for example, `session3.html`.
  - This ensures the HTML file generated has the correct name.
  
- **Include the custom stylesheet**:  
  To apply the correct styling, reference the `styles.css` file, which is located in the parent directory:
  ```yaml
  format:
    html:
      css: ../styles.css  # Reference the CSS file located in the parent folder
  ```

This makes sure the HTML file will be styled consistently with the site.

### To add sessions (or other things) to the website pages:

1. Go to the associated `.qmd` file (`index.qmd` for the main page, `sessions.qmd` for the sessions page, or `links.qmd` for the links page).
2. Add the link you want to the `.qmd` document using standard markdown or HTML links.

For example, if you want to link to `session3.html` from `sessions.qmd`, add:
```markdown
- [Session 3](session3.html)
```

3. Run the `renderSite.py` script to render the site.  
4. To make sure everything looks good, you can open `index.html` from the `docs` subfolder.
5. If everything looks good, make a pull request.

---

# Extra information for making things look good

Within your `.qmd` file, you can use HTML `div` tags to make cool effects. Here are a couple of things you can do:

### 1. Create "Terminal" Style Blocks

If you want to create a block that looks like a terminal, you can use a custom `div` with a specific class. Here’s how to do it:

```html
<div class="terminal">
  <pre><code>
  $ python script.py
  Output: Hello, World!
  </code></pre>
</div>
```

- The `<div class="terminal">` sets up the block to look like a terminal (assuming you have terminal-style CSS).
- The `<pre><code>` inside the div ensures that the content appears preformatted, like in a real terminal window.

To ensure this works, your `../styles.css` file should have the following:

```css
.terminal {
  background-color: #1e1e1e;
  color: #ffffff;
  padding: 10px;
  font-family: monospace;
  border-radius: 5px;
  overflow: auto;
}
```

### 2. Create Hideable Information with a "Details" Block

If you want to create collapsible/expandable content (useful for optional or additional information), you can use the `<details>` and `<summary>` HTML tags. Here’s how:

```html
<details>
  <summary>Click to see more details</summary>
  <p>Here is the additional information you can show or hide.</p>
</details>
```

- `<details>` creates a collapsible block.
- `<summary>` is the clickable text that expands or collapses the block.
- The content inside the `<details>` will be hidden until the user clicks to expand it.

This is great for optional information, tips, or explanations you don’t want to take up too much space on the page.

