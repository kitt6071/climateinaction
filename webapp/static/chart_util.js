(function() {
    'use strict';

    if (typeof window.AppState === 'undefined' || typeof window.AppUtils === 'undefined') {
        console.error('chart_util.js: Global state not ready');
        return;
    }

    let topThreatsChartInstance = null;
    let speciesThreatCountChartInstance = null;

    function createBarChart(ctx, chartInstance, chartData) {
        if (chartInstance) chartInstance.destroy();

        const { labels, dataValues, label, backgroundColor, borderColor } = chartData;
        const maxValue = Math.max(...dataValues);

        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: dataValues,
                    backgroundColor: backgroundColor,
                    borderColor: borderColor,
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        suggestedMax: maxValue + Math.ceil(maxValue * 0.1) + 1
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) { label += ': '; }
                                if (context.parsed.x !== null) { label += context.parsed.x; }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    function renderTopThreatsChart(data, ctx) {
        const threatCategoryCounts = data.reduce((acc, triplet) => {
            const category = triplet.object || "Unknown Category";
            acc[category] = (acc[category] || 0) + 1;
            return acc;
        }, {});

        const sortedThreatCategories = Object.entries(threatCategoryCounts)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5);

        if (ctx && sortedThreatCategories.length > 0) {
            const chartData = {
                labels: sortedThreatCategories.map(entry => entry[0]),
                dataValues: sortedThreatCategories.map(entry => entry[1]),
                label: 'Occurrences',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgba(255, 99, 132, 1)'
            };
            topThreatsChartInstance = createBarChart(ctx, topThreatsChartInstance, chartData);
            window.AppState.charts.topThreatsChart = topThreatsChartInstance;
        } else if (ctx) {
            ctx.canvas.parentElement.innerHTML = '<p>No data for threats chart.</p>';
        }
    }

    function renderSpeciesThreatCountChart(data, ctx) {
        const speciesThreats = data.reduce((acc, triplet) => {
            const species = triplet.subject || "Unknown Species";
            const threatIdentifier = `${triplet.predicate || ''} | ${triplet.object || ''}`;
            if (!acc[species]) acc[species] = new Set();
            acc[species].add(threatIdentifier);
            return acc;
        }, {});

        const speciesThreatCountsArray = Object.entries(speciesThreats)
            .map(([species, threatsSet]) => [species, threatsSet.size])
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5);

        if (ctx && speciesThreatCountsArray.length > 0) {
            const chartData = {
                labels: speciesThreatCountsArray.map(entry => entry[0]),
                dataValues: speciesThreatCountsArray.map(entry => entry[1]),
                label: 'Distinct Threats',
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)'
            };

            speciesThreatCountChartInstance = createBarChart(ctx, speciesThreatCountChartInstance, chartData);
            window.AppState.charts.speciesThreatCountChart = speciesThreatCountChartInstance;
        } else if (ctx) {
            ctx.canvas.parentElement.innerHTML = '<p>No data for species chart.</p>';
        }
    }

    function processAndRenderCharts(data) {
        if (!data || data.length === 0) return;

        if (typeof Chart === 'undefined') {
            console.error("chart_util.js: Chart.js not loaded");
            const topThreatsChartCtx = window.AppUtils.getElement('topThreatsChartCtx');
            const speciesThreatCountChartCtx = window.AppUtils.getElement('speciesThreatCountChartCtx');
            if (topThreatsChartCtx) topThreatsChartCtx.canvas.parentElement.innerHTML = '<p>Chart library missing.</p>';
            if (speciesThreatCountChartCtx) speciesThreatCountChartCtx.canvas.parentElement.innerHTML = '<p>Chart library missing.</p>';
            return;
        }

        const topThreatsChartCtx = window.AppUtils.getElement('topThreatsChartCtx');
        const speciesThreatCountChartCtx = window.AppUtils.getElement('speciesThreatCountChartCtx');

        renderTopThreatsChart(data, topThreatsChartCtx);
        renderSpeciesThreatCountChart(data, speciesThreatCountChartCtx);

        console.log('chart_util.js: charts rendered');
    }

    function getTripletById(id) {
        if (!window.AppState.allTripletsData || typeof id === 'undefined') return null;
        return window.AppState.allTripletsData.find(t => t.id === id);
    }

    function calculateTextSimilarity(text1, text2) {
        if (!text1 || !text2) return 0;

        const words1 = new Set(text1.toLowerCase().split(/\s+/));
        const words2 = new Set(text2.toLowerCase().split(/\s+/));

        const intersection = new Set([...words1].filter(word => words2.has(word)));
        const union = new Set([...words1, ...words2]);

        return intersection.size / union.size;
    }

    window.processAndRenderCharts = processAndRenderCharts;
    window.getTripletById = getTripletById;
    window.calculateTextSimilarity = calculateTextSimilarity;

    if (window.AppFunctions) {
        Object.assign(window.AppFunctions, {
            processAndRenderCharts,
            getTripletById,
            calculateTextSimilarity
        });
    }

    console.log('chart_util.js: module loaded');

})();