/**
 * Car Resale Price Valuation Tool - Client Script
 */

document.addEventListener("DOMContentLoaded", function () {
    const brandSelect = document.getElementById("brandSelect");
    const modelSelect = document.getElementById("modelSelect");
    const modelLoadingSpinner = document.getElementById("modelLoadingSpinner");

    const mileageInput = document.getElementById("mileageInput");
    const engineInput = document.getElementById("engineInput");
    const powerInput = document.getElementById("powerInput");
    const seatsSelect = document.getElementById("seatsSelect");

    const valuationForm = document.getElementById("valuationForm");
    const predictBtn = document.getElementById("predictBtn");
    const btnIcon = document.getElementById("btnIcon");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");

    const resultContainer = document.getElementById("resultContainer");
    const valPrice = document.getElementById("valPrice");
    const valRange = document.getElementById("valRange");
    const valSubtitle = document.getElementById("valSubtitle");

    const sumAge = document.getElementById("sumAge");
    const sumYear = document.getElementById("sumYear");
    const sumUsage = document.getElementById("sumUsage");
    const sumUsageStatus = document.getElementById("sumUsageStatus");
    const sumFuelTrans = document.getElementById("sumFuelTrans");
    const sumTrans = document.getElementById("sumTrans");
    const sumCondition = document.getElementById("sumCondition");
    const sumOwner = document.getElementById("sumOwner");

    const insightBrand = document.getElementById("insightBrand");
    const insightOwner = document.getElementById("insightOwner");
    const insightUsage = document.getElementById("insightUsage");

    // 1. KM Preset Quick Buttons
    document.querySelectorAll(".km-preset").forEach(function (button) {
        button.addEventListener("click", function () {
            const km = this.getAttribute("data-km");
            const kmInput = document.getElementById("kmInput");
            if (kmInput) {
                kmInput.value = km;
            }
        });
    });

    // 2. Fetch Models when Brand changes
    async function updateModelsForBrand(brand) {
        if (!brand) return;
        if (modelLoadingSpinner) modelLoadingSpinner.classList.remove("d-none");

        try {
            const response = await fetch(`/api/models?brand=${encodeURIComponent(brand)}`);
            const data = await response.json();

            if (data.status === "success" && data.models && data.models.length > 0) {
                modelSelect.innerHTML = "";
                data.models.forEach(function (m) {
                    const opt = document.createElement("option");
                    opt.value = m;
                    opt.textContent = m;
                    modelSelect.appendChild(opt);
                });

                // Auto-fetch specs for the newly selected first model
                await updateSpecsForCar(brand, data.models[0]);
            }
        } catch (err) {
            console.error("Error fetching models:", err);
        } finally {
            if (modelLoadingSpinner) modelLoadingSpinner.classList.add("d-none");
        }
    }

    // 3. Fetch default specs for Brand & Model
    async function updateSpecsForCar(brand, modelName) {
        if (!brand || !modelName) return;

        try {
            const response = await fetch(`/api/specs?brand=${encodeURIComponent(brand)}&model=${encodeURIComponent(modelName)}`);
            const data = await response.json();

            if (data.status === "success" && data.specs) {
                if (mileageInput) mileageInput.value = data.specs.mileage_kmpl;
                if (engineInput) engineInput.value = data.specs.engine_cc;
                if (powerInput) powerInput.value = data.specs.max_power_bhp;
                if (seatsSelect) {
                    seatsSelect.value = data.specs.seats;
                }
            }
        } catch (err) {
            console.error("Error fetching specs:", err);
        }
    }

    if (brandSelect) {
        brandSelect.addEventListener("change", function () {
            updateModelsForBrand(this.value);
        });
    }

    if (modelSelect) {
        modelSelect.addEventListener("change", function () {
            if (brandSelect) {
                updateSpecsForCar(brandSelect.value, this.value);
            }
        });
    }

    // 4. Form Submission via AJAX (with graceful fallback to regular POST if JS fails)
    if (valuationForm) {
        valuationForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            // Set loading state on button
            predictBtn.disabled = true;
            btnIcon.classList.add("d-none");
            btnSpinner.classList.remove("d-none");
            btnText.textContent = "Calculating Resale Value...";

            const formData = new FormData(valuationForm);
            const payload = {};
            formData.forEach(function (value, key) {
                payload[key] = value;
            });

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-Requested-With": "XMLHttpRequest"
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (response.ok && data.status === "success" && data.result) {
                    const res = data.result;

                    // Update result banner
                    valPrice.textContent = res.formatted_price;
                    valRange.textContent = res.formatted_range;
                    valSubtitle.textContent = `(Accurate estimate for ${res.brand} ${res.model} in ${res.city})`;

                    // Update summary cards
                    sumAge.textContent = `${res.car_age} Years`;
                    sumYear.textContent = `Mfg. ${res.manufactured_year}`;
                    sumUsage.textContent = `${res.avg_yearly_km.toLocaleString()} km/yr`;
                    sumUsageStatus.textContent = res.usage_status;
                    sumFuelTrans.textContent = res.fuel;
                    sumTrans.textContent = res.transmission;
                    sumCondition.textContent = res.condition;
                    sumOwner.textContent = res.owner;

                    // Update insights
                    insightBrand.textContent = res.insights.brand_retention;
                    insightOwner.textContent = res.insights.ownership_impact;
                    insightUsage.textContent = res.insights.usage_level;

                    // Show result container
                    resultContainer.classList.remove("d-none");

                    // Smooth scroll to result
                    resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
                } else {
                    alert("Valuation calculation error: " + (data.message || "Unknown error"));
                }
            } catch (err) {
                console.error("Valuation request error:", err);
                alert("Failed to calculate car valuation. Please verify your connection.");
            } finally {
                // Restore button state
                predictBtn.disabled = false;
                btnSpinner.classList.add("d-none");
                btnIcon.classList.remove("d-none");
                btnText.textContent = "Calculate Estimated Resale Value";
            }
        });
    }
});
