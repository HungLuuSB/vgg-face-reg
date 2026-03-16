// --- IEEE Formatting Function ---
#let ieee-paper(title: "", authors: (), abstract: none, keywords: (), body) = {
  // Document metadata
  set document(title: title, author: authors.map(a => a.name))

  // Page setup: A4, standard IEEE margins
  set page(
    paper: "a4",
    margin: (left: 1.5cm, right: 1.5cm, top: 2cm, bottom: 2cm),
  )

  // Text setup: Times New Roman, 11pt, justified per your requirements
  set text(font: "Times New Roman", size: 11pt)
  set par(justify: true, leading: 0.55em)

  // Heading configurations
  set heading(numbering: "I.A.1.")
  show heading: it => {
    set text(size: 11pt, weight: "bold")
    if it.level == 1 {
      set align(center)
      smallcaps(it)
    } else {
      set align(left)
      it
    }
    v(0.75em)
  }

  // Render Title
  align(center, text(18pt, weight: "bold", title))
  v(1.5em)

  // Render Authors in a flexible grid
  align(center, grid(
    columns: (1fr, 1fr, 1fr), // 3 columns for 5 authors (3 top, 2 bottom)
    row-gutter: 2em,
    ..authors.map(author => [
      *#author.name*\
      Student ID: #author.id\
      //Ho Chi Minh City University of Foreign Languages and Information Technology\
      #if "email" in author [ #link("mailto:" + author.email)[#author.email] ]
    ])
  ))
  v(2.5em)

  // Two-column layout for the main body
  show: columns.with(2, gutter: 1em)

  // Render Abstract and Keywords
  if abstract != none [
    *Abstract*---#abstract

    #v(0.5em)
    *Keywords*---#keywords.join(", ")

    #v(1.5em)
  ]

  body
}

// --- Document Content ---
#show: ieee-paper.with(
  title: [Open-Set Face Verification Using VGGFace Embeddings and MTCNN Alignment],
  authors: (
    (name: "Lưu Thái Hưng", id: "23DH111389", email: "23dh111389@st.huflit.edu.vn"),
    (name: "Lê Đỗ Huy Hùng", id: "23DH111359"),
    (name: "Lê Văn Hào Kiệt", id: "23DH114418"),
    (name: "Nguyễn Văn Ba", id: "23DH000000"),
    (name: "Lê Ngô Kim Yến", id: "23DH110000"),
  ),
  abstract: [
    This paper presents a robust, open-set face verification system designed for real-time application. By shifting from a closed-set softmax classification approach to metric learning, we leverage pre-trained VGGFace embeddings to effectively distinguish between known individuals and unknown subjects. Multitask Cascaded Convolutional Networks (MTCNN) are employed for precise facial detection and landmark-based geometric alignment. We construct a reference gallery using centroid embeddings and evaluate system performance using Cosine Similarity thresholds derived from an empirical evaluation of Genuine and Imposter image pairs.
  ],
  keywords: ("Deep Learning", "Face Recognition", "VGGFace", "MTCNN", "Metric Learning", "Open-Set Verification"),
)

= Introduction
Face recognition in unconstrained environments remains a significant challenge in computer vision. While closed-set classification models perform well on static datasets, they fail to mathematically isolate unknown identities in real-world scenarios. This paper proposes...

= Methodology
== Data Acquisition and Preprocessing
To capture sufficient intra-class variance, we extract 800 frames per subject across varying lighting conditions and poses. MTCNN is utilized to extract bounding boxes and 5-point facial landmarks for geometric normalization.

== Feature Extraction via VGGFace
Instead of employing a 6-class softmax layer, we bypass the classification head and utilize the frozen feature layers of VGGFace to map aligned faces into a high-dimensional continuous feature space.

= System Evaluation
== Threshold Determination
We evaluate the system using a paired testing protocol, calculating the Cosine Similarity between live embeddings and stored gallery centroids...

= Conclusion
// Conclusion text here

= References
// Use Typst's built-in #bibliography() function here linked to your .bib file.
