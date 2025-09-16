
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { createClient } = require("@supabase/supabase-js");

const app = express();
app.use(cors());
app.use(bodyParser.json());

const supabase = createClient(
  "https://stkjouawzqjghqryiibs.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN0a2pvdWF3enFqZ2hxcnlpaWJzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwMzY5NTIsImV4cCI6MjA3MzYxMjk1Mn0.39okca3yiiiOMwKP7uuJhyJdwCY1KJz49EWvlCxpJ6M"
);

app.post("/api/register", async (req, res) => {
  const { fullName, nick, email } = req.body;
  const ip = req.headers["x-forwarded-for"] || req.socket.remoteAddress;
  const date = new Date().toISOString();

  try {
    const { error } = await supabase
      .from("accepted")
      .insert([{ fullName, nick, email, ip, created_at: date }]);
    if (error) throw error;
    res.json({ message: "Rejestracja zapisana" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Błąd serwera" });
  }
});

app.listen(3000, () => console.log("Serwer działa na http://localhost:3000"));
