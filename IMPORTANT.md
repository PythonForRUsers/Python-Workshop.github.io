# Important information for session development!

If you're making a session:
Put all needed files in a folder titled "session#" where # is the session number.  
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

---

# How to Use the Custom Styles from `styles.css`

### 1. **Creating "Terminal" Style Blocks**

To create a block that mimics the appearance of a terminal, use the `terminal` class. This is useful for displaying code or terminal commands in a styled, preformatted block.

#### Example:
```html
<div class="terminal">
  $ python script.py
  Output: Hello, World!
</div>
```

- **Appearance**: This block will have a dark background and monospace font, resembling a terminal window.
- **When to use**: Use this for displaying terminal commands or code snippets in a stylized format.

### 2. **Creating "Nice Blocks" for Tips or Notes**

To create a block that has a light background, use the `niceblock` class. This can be used for highlighting important tips, notes, or warnings.

#### Example:
```html
<div class="niceblock">
  <p>This is a helpful tip or note!</p>
</div>
```

- **Appearance**: The block will have a light blue background with a border and rounded corners, perfect for highlighting key information.
- **When to use**: Use this for highlighting tips, suggestions, or important reminders in your content.

### 3. **Creating Link Blocks with Images and Centered Text**

If you want to create a clickable block that contains an image and centered text underneath the image, use the `link-block` class.

#### Example:
```html
<a href="https://example.com" class="link-block">
    <img src="path-to-image.jpg" alt="Image description">
    <p>Click here for more information</p>
</a>
```

- **Appearance**: This creates a clickable block where an image is displayed, and the text is centered beneath it.
- **When to use**: Use this when you want to create a link that includes both an image and text, like a button or a visual call-to-action.

### 4. **Creating Hideable Information with a "Details" Block**

The `details` and `summary` elements allow you to create expandable/collapsible content. Use this to create sections that the user can expand to reveal more information.

#### Example:
```html
<details>
  <summary>Click to see more details</summary>
  <p>Here is some additional information that can be shown or hidden.</p>
</details>
```

- **Appearance**: This creates a light blue block that the user can expand and collapse. The summary text will be bold, with a hover effect to indicate it's clickable.
- **When to use**: Use this when you want to hide additional details or optional information that doesn’t need to be visible at first.

### 5. **Styling Links**

The `styles.css` file also defines styles for links. You don’t need to use any special class—these styles will apply to all links automatically.

#### Link Styles:
- **Normal state**: Links will appear in a dark blue color (`--links`).
- **Hover state**: When a user hovers over a link, it will change to a lighter blue (`--hover`).
- **Active state**: When a link is clicked, the color will remain the same as the hover state.
- **Visited state**: Visited links will change to the hover color.

#### Example:
```markdown
[Click here to visit the website](https://example.com)
```

This link will automatically inherit the styles defined in the CSS file.

---

