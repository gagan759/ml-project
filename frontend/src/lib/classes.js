import { Bug, Droplets, FlaskConical, Leaf, Sparkles, Sprout, Sun, Waves } from "lucide-react";

// Fixed class order used by every model, the API and the UI (alphabetical by dataset folder).
export const CLASSES = [
  { key: "early_blight", name: "Early Blight", short: "Early blight", abbr: "EB", color: "#c2410c", category: "Fungal", icon: Sun },
  { key: "healthy", name: "Healthy", short: "Healthy", abbr: "H", color: "#16a34a", category: "Healthy", icon: Leaf },
  { key: "late_blight", name: "Late Blight", short: "Late blight", abbr: "LB", color: "#7c3aed", category: "Oomycete", icon: Droplets },
  { key: "leaf_miner", name: "Leaf Miner", short: "Leaf miner", abbr: "LM", color: "#0891b2", category: "Insect pest", icon: Bug },
  { key: "magnesium_deficiency", name: "Magnesium Deficiency", short: "Magnesium", abbr: "Mg", color: "#ca8a04", category: "Nutrient deficiency", icon: FlaskConical },
  { key: "nitrogen_deficiency", name: "Nitrogen Deficiency", short: "Nitrogen", abbr: "N", color: "#65a30d", category: "Nutrient deficiency", icon: Sprout },
  { key: "potassium_deficiency", name: "Potassium Deficiency", short: "Potassium", abbr: "K", color: "#ea580c", category: "Nutrient deficiency", icon: Sparkles },
  { key: "spotted_wilt_virus", name: "Spotted Wilt Virus", short: "Spotted wilt", abbr: "SWV", color: "#db2777", category: "Viral", icon: Waves },
];

export const CLASS_BY_KEY = Object.fromEntries(CLASSES.map((c) => [c.key, c]));
export const classMeta = (key) => CLASS_BY_KEY[key] || { key, name: key, short: key, abbr: "?", color: "#64748b", icon: Leaf };

export const STATUS_STYLE = {
  Healthy: { tone: "leaf", label: "Healthy", description: "No disease, pest or deficiency detected." },
  Diseased: { tone: "tomato", label: "Condition detected", description: "The model recognised a condition with confidence above its threshold." },
  Uncertain: { tone: "sun", label: "Uncertain", description: "Confidence is below the model's validated threshold, so no diagnosis is given." },
  "Unusual image": { tone: "slate", label: "Unusual image", description: "This photo looks unlike every training photo (a tomato leaf on paper). No diagnosis is given." },
};
